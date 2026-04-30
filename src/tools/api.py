from typing import Dict, Any, List
import os
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import json
import numpy as np
from src.utils.logging_config import setup_logger

# 设置日志记录
logger = setup_logger('api')


def get_stock_name(symbol: str) -> str:
    """根据股票代码获取股票名称"""
    cache_file = "src/data/stock_name_cache.json"
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if symbol in cache:
                return cache[symbol].get('name', '')
    except:
        pass
    
    try:
        stock_info = ak.stock_info_a_code_name()
        if stock_info is not None and not stock_info.empty:
            match = stock_info[stock_info['code'] == symbol]
            if not match.empty:
                name = str(match.iloc[0].get('name', ''))
                if name:
                    try:
                        cache = {}
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                cache = json.load(f)
                        cache[symbol] = {"name": name, "updated": datetime.now().isoformat()}
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(cache, f, ensure_ascii=False)
                    except:
                        pass
                    return name
    except Exception as e:
        logger.debug(f"获取股票名称失败: {e}")
    return ''


def _get_stock_prefix(symbol: str) -> str:
    """根据股票代码判断交易所前缀"""
    ticker_int = int(symbol)
    # 沪市: 600000-603999, 688000-688999
    # 深市: 000001-003999, 300000-300999
    if 600000 <= ticker_int <= 603999 or 688000 <= ticker_int <= 688999:
        return f"sh{symbol}"
    else:  # 深市
        return f"sz{symbol}"


def _is_expected_network_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    markers = [
        "nameresolutionerror",
        "failed to resolve",
        "nodename nor servname provided",
        "proxyerror",
        "remote end closed connection",
        "connection aborted",
        "connectionpool",
        "ssl:",
        "ssleoferror",
        "no value to decode",
    ]
    return any(marker in error_msg for marker in markers)


def _log_data_source_failure(action: str, error: Exception) -> None:
    if _is_expected_network_error(error):
        logger.warning("%s unavailable, using fallback data: %s", action, error)
    else:
        logger.error("%s failed: %s", action, error)


def _get_latest_price_from_tx(symbol: str) -> float:
    """从腾讯历史行情获取最新收盘价作为备选方案"""
    try:
        stock_code = f"sz{symbol}" if symbol.startswith(("0", "3")) else f"sh{symbol}"
        df = ak.stock_zh_a_hist_tx(symbol=stock_code)
        if df is not None and not df.empty:
            latest_close = float(df['close'].iloc[-1])
            logger.info(f"✓ 从腾讯历史行情获取最新价格: {latest_close}")
            return latest_close
    except Exception as e:
        _log_data_source_failure("Tencent price history", e)
    return 0


def _estimate_market_cap_from_financials(symbol: str, price: float) -> float:
    """使用新浪财报中的股本信息粗略估算市值，避免依赖东方财富接口。"""
    cache_file = "src/data/market_cap_cache.json"

    cached_price = 0
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if symbol in cache:
                    cached_price = cache[symbol].get("price", 0)
    except Exception as e:
        logger.warning(f"读取缓存价格失败: {e}")

    if cached_price > 0:
        price = cached_price

    if price <= 0:
        price = _get_latest_price_from_tx(symbol)
        if price <= 0:
            return 0

    try:
        stock_prefix = "sz" if symbol.startswith(("0", "3")) else "sh"
        balance = ak.stock_financial_report_sina(stock=f"{stock_prefix}{symbol}", symbol="资产负债表")
        if balance is None or balance.empty:
            return 0
        total_shares = 0
        for col in balance.columns:
            if "股本" in str(col) or "资本" in str(col):
                raw_value = balance[col].iloc[0]
                total_shares = float(raw_value) if not pd.isna(raw_value) else 0
                if total_shares > 0:
                    break
        return price * total_shares if total_shares > 0 else 0
    except Exception as e:
        _log_data_source_failure("Market cap fallback calculation", e)
        return 0


def get_financial_metrics(symbol: str) -> Dict[str, Any]:
    """获取财务指标数据"""
    logger.info(f"Getting financial indicators for {symbol}...")

    stock_data = pd.Series()
    latest_financial = pd.Series()
    latest_income = pd.Series()

    # 检查财务指标缓存
    financial_cache_file = "src/data/financial_metrics_cache.json"
    try:
        if os.path.exists(financial_cache_file):
            with open(financial_cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if symbol in cache:
                    cached_data = cache[symbol]
                    # 检查缓存是否在当天
                    cache_time = datetime.fromisoformat(cached_data.get("updated", "2000-01-01"))
                    if cache_time.date() == datetime.now().date():
                        logger.info(f"✓ 使用缓存的财务指标数据")
                        return cached_data.get("metrics", {})
    except Exception as e:
        logger.warning(f"读取财务指标缓存失败: {e}")

    cache_file = "src/data/market_cap_cache.json"
    cached_price = 0

    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if symbol in cache:
                    cached_price = cache[symbol].get("price", 0)
    except Exception as e:
        logger.warning(f"读取价格缓存失败: {e}")

    if cached_price > 0:
        logger.info(f"✓ 使用缓存的价格: {cached_price}")
    else:
        logger.info("跳过实时行情API，尝试使用其他方式获取价格")

    # 获取财务分析指标 - 先尝试sina，失败则用东方财富备用
    # 同时获取年报和最新季报数据
    try:
        logger.info("Fetching Sina financial indicators...")
        current_year = datetime.now().year
        financial_data = ak.stock_financial_analysis_indicator(
            symbol=symbol, start_year=str(current_year-2))
        if financial_data is not None and not financial_data.empty:
            financial_data['日期'] = pd.to_datetime(financial_data['日期'])
            financial_data = financial_data.sort_values('日期', ascending=False)

            # 获取年报数据（12月31日）
            annual_reports = financial_data[financial_data['日期'].dt.month == 12]
            # 获取最新报告数据（可能是季报或年报）
            latest_report = financial_data.iloc[0]

            if not annual_reports.empty:
                annual_report = annual_reports.iloc[0]
                logger.info(f"✓ 年报数据: {annual_report['日期'].strftime('%Y-%m-%d')}, ROE: {annual_report.get('净资产收益率(%)', 'N/A')}%")
                # 如果最新报告不是年报，同时记录季报信息
                if latest_report['日期'] != annual_report['日期']:
                    logger.info(f"✓ 最新季报数据: {latest_report['日期'].strftime('%Y-%m-%d')}, ROE: {latest_report.get('净资产收益率(%)', 'N/A')}%")
                    # 将季报数据也保存，供后续使用
                    latest_financial = latest_report.copy()
                    # 用年报数据补充一些关键指标（如全年增长率）
                    for col in ['净利润增长率(%)', '主营业务收入增长率(%)', '净资产增长率(%)']:
                        if pd.isna(latest_financial.get(col)) or latest_financial.get(col) == 0:
                            if not pd.isna(annual_report.get(col)):
                                latest_financial[col] = annual_report[col]
                else:
                    latest_financial = annual_report
            else:
                latest_financial = latest_report
                logger.info(f"✓ Financial indicators fetched ({len(financial_data)} records)")
    except Exception as e:
        logger.warning(f"Failed to get Sina financial indicators: {e}")
        try:
            logger.info("Trying Eastmoney financial abstract as fallback...")
            fin_abstract = ak.stock_financial_abstract(symbol=symbol)
            if fin_abstract is not None and not fin_abstract.empty:
                col_2025 = str(current_year) + "1231"
                col_2024 = str(current_year-1) + "1231"
                latest_cols = [col for col in fin_abstract.columns if col in [col_2025, col_2024, "20251231", "20241231", "20231231"]]
                latest_col = latest_cols[0] if latest_cols else fin_abstract.columns[2] if len(fin_abstract.columns) > 2 else None
                if latest_col:
                    latest_financial = pd.Series(dict(zip(fin_abstract['指标'], fin_abstract[latest_col])))
                    latest_financial['净资产收益率(%)'] = latest_financial.get('归母净资产收益率') * 100 if pd.notna(latest_financial.get('归母净资产收益率')) else 0
                    latest_financial['销售净利率(%)'] = latest_financial.get('销售净利率') * 100 if pd.notna(latest_financial.get('销售净利率')) else 0
                    latest_financial['每股净资产_调整前(元)'] = latest_financial.get('归母每股净资产')
                    latest_financial['加权每股收益(元)'] = latest_financial.get('扣非每股收益')
                    logger.info(f"✓ Financial data from Eastmoney, date: {latest_col}")
        except Exception as e2:
            logger.warning(f"Eastmoney fallback also failed: {e2}")

    if latest_financial.empty:
        logger.warning("No financial indicator data available")

    # 获取利润表数据 - 备用尝试东方财富
    stock_prefix = _get_stock_prefix(symbol)
    try:
        logger.info("Fetching income statement...")
        income_statement = ak.stock_financial_report_sina(
            stock=stock_prefix, symbol="利润表")
        if income_statement is not None and not income_statement.empty:
            latest_income = income_statement.iloc[0]
            logger.info("✓ Income statement fetched")
    except Exception as e:
        logger.warning(f"Failed to get income statement: {e}")
        try:
            fin_abstract = ak.stock_financial_abstract(symbol=symbol)
            if fin_abstract is not None and not fin_abstract.empty:
                col_2025 = str(datetime.now().year) + "1231"
                col_2024 = str(datetime.now().year-1) + "1231"
                latest_cols = [col for col in fin_abstract.columns if col in [col_2025, col_2024, "20251231", "20241231", "20231231"]]
                latest_col = latest_cols[0] if latest_cols else fin_abstract.columns[2] if len(fin_abstract.columns) > 2 else None
                if latest_col:
                    latest_income = pd.Series(dict(zip(fin_abstract['指标'], fin_abstract[latest_col])))
                    logger.info("✓ Income data from Eastmoney fallback")
        except Exception as e2:
            logger.warning(f"Eastmoney income fallback also failed: {e2}")

    # 构建完整指标数据
    logger.info("Building indicators...")
    try:
        def convert_percentage(value: float) -> float:
            """将百分比值转换为小数
            如果|值|>1，说明已经是百分比形式(如34.21表示34.21%)，需要除以100
            如果|值|<=1，说明已经是小数形式(如0.3421表示34.21%)，直接返回
            """
            try:
                fv = float(value) if value is not None else 0.0
                if abs(fv) > 1:
                    return fv / 100.0
                return fv
            except:
                return 0.0

        all_metrics = {
            # 市场数据 - 从个股信息获取市值
            "market_cap": 0,  # 将在后面从个股信息获取
            "float_market_cap": 0,

            # 盈利数据
            "revenue": float(latest_income.get("营业总收入", 0)) if not latest_income.empty else 0,
            "net_income": float(latest_income.get("净利润", 0)) if not latest_income.empty else 0,
            "return_on_equity": convert_percentage(latest_financial.get("净资产收益率(%)", 0)) if not latest_financial.empty else 0,
            "net_margin": convert_percentage(latest_financial.get("销售净利率(%)", 0)) if not latest_financial.empty else 0,
            "operating_margin": convert_percentage(latest_financial.get("营业利润率(%)", 0)) if not latest_financial.empty else 0,

            # 增长指标
            "revenue_growth": convert_percentage(latest_financial.get("主营业务收入增长率(%)", 0)) if not latest_financial.empty else 0,
            "earnings_growth": convert_percentage(latest_financial.get("净利润增长率(%)", 0)) if not latest_financial.empty else 0,
            "book_value_growth": convert_percentage(latest_financial.get("净资产增长率(%)", 0)) if not latest_financial.empty else 0,

            # 财务健康指标
            "current_ratio": float(latest_financial.get("流动比率", 0)) if not latest_financial.empty else 0,
            "debt_to_equity": convert_percentage(latest_financial.get("资产负债率(%)", 0)) if not latest_financial.empty else 0,
            "free_cash_flow_per_share": float(latest_financial.get("每股经营性现金流(元)", 0)) if not latest_financial.empty else 0,
            "earnings_per_share": float(latest_financial.get("加权每股收益(元)", 0)) if not latest_financial.empty else 0,

            # 估值比率 - 从财务数据计算
            "pe_ratio": 0,
            "price_to_book": 0,
            "price_to_sales": 0,
        }

        price = float(stock_data.get("最新价", 0)) if not stock_data.empty else 0
        if price <= 0:
            price = _get_latest_price_from_tx(symbol)

        market_cap = _estimate_market_cap_from_financials(symbol, price)
        all_metrics["market_cap"] = market_cap

        book_value_per_share = float(latest_financial.get("每股净资产_调整前(元)", 0)) if not latest_financial.empty else 0
        eps = float(latest_financial.get("加权每股收益(元)", 0)) if not latest_financial.empty else 0
        total_revenue = float(latest_income.get("营业总收入", 0)) if not latest_income.empty else 0

        if price > 0:
            if book_value_per_share > 0:
                all_metrics["price_to_book"] = round(price / book_value_per_share, 2)
            if eps > 0:
                all_metrics["pe_ratio"] = round(price / eps, 2)
            if total_revenue > 0 and market_cap > 0:
                all_metrics["price_to_sales"] = round(market_cap / total_revenue, 2)
        if all_metrics["market_cap"] > 0:
            logger.info(f"✓ 通过新浪财报估算市值: {all_metrics['market_cap']}")

        # 只返回 agent 需要的指标
        agent_metrics = {
            # 盈利能力指标
            "return_on_equity": all_metrics["return_on_equity"],
            "net_margin": all_metrics["net_margin"],
            "operating_margin": all_metrics["operating_margin"],

            # 增长指标
            "revenue_growth": all_metrics["revenue_growth"],
            "earnings_growth": all_metrics["earnings_growth"],
            "book_value_growth": all_metrics["book_value_growth"],

            # 财务健康指标
            "current_ratio": all_metrics["current_ratio"],
            "debt_to_equity": all_metrics["debt_to_equity"],
            "free_cash_flow_per_share": all_metrics["free_cash_flow_per_share"],
            "earnings_per_share": all_metrics["earnings_per_share"],

            # 估值比率
            "pe_ratio": all_metrics["pe_ratio"],
            "price_to_book": all_metrics["price_to_book"],
            "price_to_sales": all_metrics["price_to_sales"],
        }

        logger.info("✓ Indicators built successfully")

        # 打印所有获取到的指标数据（用于调试）
        logger.debug("\n获取到的完整指标数据：")
        for key, value in all_metrics.items():
            logger.debug(f"{key}: {value}")

        logger.debug("\n传递给 agent 的指标数据：")
        for key, value in agent_metrics.items():
            logger.debug(f"{key}: {value}")

        # 如果指标全为0，说明没有获取到有效数据
        if all(v == 0 for v in agent_metrics.values()):
            logger.warning("All metrics are zero, returning empty")
            return [{}]

        # 写入缓存
        try:
            cache = {}
            if os.path.exists(financial_cache_file):
                with open(financial_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            cache[symbol] = {
                "metrics": agent_metrics,
                "updated": datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(financial_cache_file), exist_ok=True)
            with open(financial_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info(f"✓ 财务指标已缓存")
        except Exception as e:
            logger.warning(f"写入财务指标缓存失败: {e}")

        return [agent_metrics]

    except Exception as e:
        logger.error(f"Error building indicators: {e}")
        return [{}]


def get_financial_statements(symbol: str) -> Dict[str, Any]:
    """获取财务报表数据"""
    logger.info(f"Getting financial statements for {symbol}...")
    stock_prefix = _get_stock_prefix(symbol)
    try:
        # 获取资产负债表数据
        logger.info("Fetching balance sheet...")
        try:
            balance_sheet = ak.stock_financial_report_sina(
                stock=stock_prefix, symbol="资产负债表")
            if not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[0]
                previous_balance = balance_sheet.iloc[1] if len(
                    balance_sheet) > 1 else balance_sheet.iloc[0]
                logger.info("✓ Balance sheet fetched")
            else:
                logger.warning("Failed to get balance sheet")
                logger.error("No balance sheet data found")
                latest_balance = pd.Series()
                previous_balance = pd.Series()
        except Exception as e:
            logger.warning("Failed to get balance sheet")
            _log_data_source_failure("Balance sheet source", e)
            latest_balance = pd.Series()
            previous_balance = pd.Series()

        # 获取利润表数据
        logger.info("Fetching income statement...")
        try:
            income_statement = ak.stock_financial_report_sina(
                stock=stock_prefix, symbol="利润表")
            if not income_statement.empty:
                latest_income = income_statement.iloc[0]
                previous_income = income_statement.iloc[1] if len(
                    income_statement) > 1 else income_statement.iloc[0]
                logger.info("✓ Income statement fetched")
            else:
                logger.warning("Failed to get income statement")
                logger.error("No income statement data found")
                latest_income = pd.Series()
                previous_income = pd.Series()
        except Exception as e:
            logger.warning("Failed to get income statement")
            _log_data_source_failure("Income statement source", e)
            latest_income = pd.Series()
            previous_income = pd.Series()

        # 获取现金流量表数据
        logger.info("Fetching cash flow statement...")
        try:
            cash_flow = ak.stock_financial_report_sina(
                stock=stock_prefix, symbol="现金流量表")
            if not cash_flow.empty:
                latest_cash_flow = cash_flow.iloc[0]
                previous_cash_flow = cash_flow.iloc[1] if len(
                    cash_flow) > 1 else cash_flow.iloc[0]
                logger.info("✓ Cash flow statement fetched")
            else:
                logger.warning("Failed to get cash flow statement")
                logger.error("No cash flow data found")
                latest_cash_flow = pd.Series()
                previous_cash_flow = pd.Series()
        except Exception as e:
            logger.warning("Failed to get cash flow statement")
            _log_data_source_failure("Cash flow statement source", e)
            latest_cash_flow = pd.Series()
            previous_cash_flow = pd.Series()

        # 构建财务数据
        line_items = []
        try:
            # 处理最新期间数据
            current_item = {
                # 从利润表获取
                "net_income": float(latest_income.get("净利润", 0)),
                "operating_revenue": float(latest_income.get("营业总收入", 0)),
                "operating_profit": float(latest_income.get("营业利润", 0)),

                # 从资产负债表计算营运资金
                "working_capital": float(latest_balance.get("流动资产合计", 0)) - float(latest_balance.get("流动负债合计", 0)),

                # 从现金流量表获取
                "depreciation_and_amortization": float(latest_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(latest_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(latest_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(current_item)
            logger.info("✓ Latest period data processed successfully")

            # 处理上一期间数据
            previous_item = {
                "net_income": float(previous_income.get("净利润", 0)),
                "operating_revenue": float(previous_income.get("营业总收入", 0)),
                "operating_profit": float(previous_income.get("营业利润", 0)),
                "working_capital": float(previous_balance.get("流动资产合计", 0)) - float(previous_balance.get("流动负债合计", 0)),
                "depreciation_and_amortization": float(previous_cash_flow.get("固定资产折旧、油气资产折耗、生产性生物资产折旧", 0)),
                "capital_expenditure": abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0))),
                "free_cash_flow": float(previous_cash_flow.get("经营活动产生的现金流量净额", 0)) - abs(float(previous_cash_flow.get("购建固定资产、无形资产和其他长期资产支付的现金", 0)))
            }
            line_items.append(previous_item)
            logger.info("✓ Previous period data processed successfully")

        except Exception as e:
            logger.error(f"Error processing financial data: {e}")
            default_item = {
                "net_income": 0,
                "operating_revenue": 0,
                "operating_profit": 0,
                "working_capital": 0,
                "depreciation_and_amortization": 0,
                "capital_expenditure": 0,
                "free_cash_flow": 0
            }
            line_items = [default_item, default_item]

        return line_items

    except Exception as e:
        _log_data_source_failure("Financial statement source", e)
        default_item = {
            "net_income": 0,
            "operating_revenue": 0,
            "operating_profit": 0,
            "working_capital": 0,
            "depreciation_and_amortization": 0,
            "capital_expenditure": 0,
            "free_cash_flow": 0
        }
        return [default_item, default_item]


def get_market_data(symbol: str) -> Dict[str, Any]:
    """获取市场数据 - 使用新浪/腾讯数据源"""
    # 初始化默认值
    volume = 0
    high_52w = 0
    low_52w = 0
    price = 0
    market_cap = 0

    # 尝试从缓存获取市值
    cache_file = "src/data/market_cap_cache.json"
    cached_market_cap = 0
    cached_price = 0

    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if symbol in cache:
                    cached_data = cache[symbol]
                    cached_market_cap = cached_data.get("market_cap", 0)
                    cached_price = cached_data.get("price", 0)
                    logger.info(f"使用缓存的市值: {cached_market_cap}, 价格: {cached_price}")
    except Exception as e:
        logger.warning(f"读取市值缓存失败: {e}")

    # 优先使用缓存的市值，避免每次都调用慢速API
    if cached_market_cap > 0:
        market_cap = cached_market_cap
        price = cached_price if cached_price > 0 else 17.0
        volume = 0
        high_52w = price * 1.1
        low_52w = price * 0.9
        logger.info(f"✓ 使用缓存的市值: {market_cap}")
    else:
        try:
            # 获取实时行情 - 新浪数据源
            realtime_data = ak.stock_zh_a_spot()
            # 新浪数据源股票代码格式
            stock_code = f"sz{symbol}" if symbol.startswith("3") or symbol.startswith("0") else f"sh{symbol}"
            stock_data = realtime_data[realtime_data['代码'] == stock_code]

            if not stock_data.empty:
                stock_data = stock_data.iloc[0]
                price = float(stock_data.get("最新价", 0))
                volume = float(stock_data.get("成交量", 0))
                high_52w = float(stock_data.get("最高", 0))
                low_52w = float(stock_data.get("最低", 0))

                # 尝试获取市值
                market_cap = _estimate_market_cap_from_financials(symbol, price)
                if market_cap > 0:
                    logger.info(f"✓ 通过新浪财报估算市值: {market_cap}")

                    # 缓存市值
                    try:
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                        cache = {}
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                cache = json.load(f)
                        cache[symbol] = {"market_cap": market_cap, "price": price, "updated": datetime.now().isoformat()}
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(cache, f, ensure_ascii=False, indent=2)
                        logger.info(f"✓ 市值已缓存")
                    except Exception as e:
                        logger.warning(f"缓存市值失败: {e}")
            else:
                price = 0
                volume = 0
                high_52w = 0
                low_52w = 0
                market_cap = 0

        except Exception as e:
            _log_data_source_failure("Market data source", e)
            price = 0
            volume = 0
            high_52w = 0
            low_52w = 0
            market_cap = 0

    # 如果市值获取失败，尝试从腾讯历史行情获取价格并估算市值
    if market_cap <= 0:
        # 先尝试使用缓存的市值
        if cached_market_cap > 0:
            market_cap = cached_market_cap
            price = cached_price
            logger.info(f"使用缓存的市值: {market_cap}")
        else:
            # 从腾讯历史行情获取最新价格
            price = _get_latest_price_from_tx(symbol)
            if price > 0:
                market_cap = _estimate_market_cap_from_financials(symbol, price)
                if market_cap > 0:
                    logger.info(f"✓ 通过腾讯历史行情估算市值: {market_cap}")
                    # 缓存市值
                    try:
                        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                        cache = {}
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                cache = json.load(f)
                        cache[symbol] = {"market_cap": market_cap, "price": price, "updated": datetime.now().isoformat()}
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(cache, f, ensure_ascii=False, indent=2)
                        logger.info(f"✓ 市值已缓存")
                    except Exception as e:
                        logger.warning(f"缓存市值失败: {e}")

    return {
        "market_cap": market_cap,
        "volume": volume,
        "average_volume": volume,
        "fifty_two_week_high": high_52w,
        "fifty_two_week_low": low_52w
    }


def get_price_history(symbol: str, start_date: str = None, end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
    """获取历史价格数据

    Args:
        symbol: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD，如果为None则默认获取过去一年的数据
        end_date: 结束日期，格式：YYYY-MM-DD，如果为None则使用昨天作为结束日期
        adjust: 复权类型，可选值：
               - "": 不复权
               - "qfq": 前复权（默认）
               - "hfq": 后复权

    Returns:
        包含以下列的DataFrame：
        - date: 日期
        - open: 开盘价
        - high: 最高价
        - low: 最低价
        - close: 收盘价
        - volume: 成交量（手）
        - amount: 成交额（元）
        - amplitude: 振幅（%）
        - pct_change: 涨跌幅（%）
        - change_amount: 涨跌额（元）
        - turnover: 换手率（%）

        技术指标：
        - momentum_1m: 1个月动量
        - momentum_3m: 3个月动量
        - momentum_6m: 6个月动量
        - volume_momentum: 成交量动量
        - historical_volatility: 历史波动率
        - volatility_regime: 波动率区间
        - volatility_z_score: 波动率Z分数
        - atr_ratio: 真实波动幅度比率
        - hurst_exponent: 赫斯特指数
        - skewness: 偏度
        - kurtosis: 峰度
    """
    try:
        # 获取当前日期和昨天的日期
        current_date = datetime.now()
        yesterday = current_date - timedelta(days=1)

        # 如果没有提供日期，默认使用昨天作为结束日期
        if not end_date:
            end_date = yesterday  # 使用昨天作为结束日期
        else:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            # 确保end_date不会超过昨天
            if end_date > yesterday:
                end_date = yesterday

        if not start_date:
            start_date = end_date - timedelta(days=365)  # 默认获取一年的数据
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")

        logger.info(f"\nGetting price history for {symbol}...")
        logger.info(f"Start date: {start_date.strftime('%Y-%m-%d')}")
        logger.info(f"End date: {end_date.strftime('%Y-%m-%d')}")

        def get_and_process_data(start_date, end_date, use_backup=False):
            """获取并处理数据，包括重命名列等操作 - 支持腾讯和东方财富数据源"""
            stock_code = f"sz{symbol}" if symbol.startswith("3") or symbol.startswith("0") else f"sh{symbol}"

            df = pd.DataFrame()

            if not use_backup:
                try:
                    df = ak.stock_zh_a_hist_tx(symbol=stock_code)
                except Exception as e:
                    logger.warning(f"腾讯数据源获取失败: {e}，尝试备用数据源...")
                    use_backup = True

            if use_backup or df.empty:
                try:
                    df = ak.stock_zh_a_hist(symbol=symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), adjust="qfq")
                except Exception as e:
                    logger.warning(f"东方财富数据源也获取失败: {e}")

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列以匹配技术分析代理的需求
            df = df.rename(columns={
                "日期": "date",
                "股票代码": "code",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude",
                "涨跌幅": "pct_change",
                "涨跌额": "change_amount",
                "换手率": "turnover",
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "amount": "amount",
            })

            # 添加缺失的列
            if "volume" not in df.columns and "amount" in df.columns:
                df["volume"] = df["amount"]

            # 添加其他需要的列（腾讯不提供，设为默认值）
            if "amount" not in df.columns:
                df["amount"] = df["volume"] * df["close"]  # 估算成交额
            if "amplitude" not in df.columns:
                df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
            if "pct_change" not in df.columns:
                df["pct_change"] = df["close"].pct_change() * 100
            if "change_amount" not in df.columns:
                df["change_amount"] = df["close"] - df["close"].shift(1)
            if "turnover" not in df.columns:
                df["turnover"] = 0  # 腾讯不直接提供换手率

            # 确保日期列为datetime类型
            df["date"] = pd.to_datetime(df["date"])

            # 过滤日期范围
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

            return df

        # 获取历史行情数据
        df = get_and_process_data(start_date, end_date)

        # 如果数据为空，使用备用数据源
        if df is None or df.empty:
            logger.warning(f"主数据源无数据，尝试备用数据源...")
            df = get_and_process_data(start_date, end_date, use_backup=True)

        if df is None or df.empty:
            logger.warning(
                f"Warning: No price history data found for {symbol}")
            return pd.DataFrame()

        # 检查数据量是否足够
        min_required_days = 120  # 至少需要120个交易日的数据
        if len(df) < min_required_days:
            logger.warning(
                f"Warning: Insufficient data ({len(df)} days) for all technical indicators")
            logger.info("Attempting to fetch more data...")

            # 扩大时间范围到2年
            start_date = end_date - timedelta(days=730)
            df = get_and_process_data(start_date, end_date)
            if df is None or df.empty:
                df = get_and_process_data(start_date, end_date, use_backup=True)

            if len(df) < min_required_days:
                logger.warning(
                    f"Warning: Even with extended time range, insufficient data ({len(df)} days)")

        # 计算动量指标
        df["momentum_1m"] = df["close"].pct_change(periods=20)  # 20个交易日约等于1个月
        df["momentum_3m"] = df["close"].pct_change(periods=60)  # 60个交易日约等于3个月
        df["momentum_6m"] = df["close"].pct_change(
            periods=120)  # 120个交易日约等于6个月

        # 计算成交量动量（相对于20日平均成交量的变化）
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()
        df["volume_momentum"] = df["volume"] / df["volume_ma20"]

        # 计算波动率指标
        # 1. 历史波动率 (20日)
        returns = df["close"].pct_change()
        df["historical_volatility"] = returns.rolling(
            window=20).std() * np.sqrt(252)  # 年化

        # 2. 波动率区间 (相对于过去120天的波动率的位置)
        volatility_120d = returns.rolling(window=120).std() * np.sqrt(252)
        vol_min = volatility_120d.rolling(window=120).min()
        vol_max = volatility_120d.rolling(window=120).max()
        vol_range = vol_max - vol_min
        df["volatility_regime"] = np.where(
            vol_range > 0,
            (df["historical_volatility"] - vol_min) / vol_range,
            0  # 当范围为0时返回0
        )

        # 3. 波动率Z分数
        vol_mean = df["historical_volatility"].rolling(window=120).mean()
        vol_std = df["historical_volatility"].rolling(window=120).std()
        df["volatility_z_score"] = (
            df["historical_volatility"] - vol_mean) / vol_std

        # 4. ATR比率
        tr = pd.DataFrame()
        tr["h-l"] = df["high"] - df["low"]
        tr["h-pc"] = abs(df["high"] - df["close"].shift(1))
        tr["l-pc"] = abs(df["low"] - df["close"].shift(1))
        tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["atr"] = tr["tr"].rolling(window=14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # 计算统计套利指标
        # 1. 赫斯特指数 (使用过去120天的数据)
        def calculate_hurst(series):
            """
            计算Hurst指数 - 使用R/S分析方法。

            Args:
                series: 价格序列

            Returns:
                float: Hurst指数，或在计算失败时返回np.nan
                   H < 0.5: 均值回归
                   H = 0.5: 随机游走
                   H > 0.5: 趋势性
            """
            try:
                series = series.dropna()
                if len(series) < 50:  # 需要足够的数据点
                    return np.nan

                # 使用对数收益率
                log_returns = np.log(series / series.shift(1)).dropna()
                if len(log_returns) < 50:
                    return np.nan

                # R/S分析计算赫斯特指数
                # 将序列分成多个子序列，计算每个子序列的R/S值
                n = len(log_returns)
                max_k = min(n // 2, 50)  # 最大子序列长度

                rs_values = []
                k_values = []

                for k in [10, 20, 30, 40, 50]:
                    if k >= n:
                        continue

                    # 将序列分成 n/k 个子序列
                    m = n // k
                    if m < 2:
                        continue

                    rs_list = []
                    for i in range(m):
                        sub_series = log_returns[i * k:(i + 1) * k]
                        if len(sub_series) < k:
                            continue

                        # 计算累积偏差
                        mean = sub_series.mean()
                        cum_dev = (sub_series - mean).cumsum()

                        # 极差 R
                        R = cum_dev.max() - cum_dev.min()

                        # 标准差 S
                        S = sub_series.std()

                        if S > 0:
                            rs_list.append(R / S)

                    if rs_list:
                        rs_values.append(np.mean(rs_list))
                        k_values.append(k)

                if len(rs_values) < 3:
                    return np.nan

                # 对数回归: log(R/S) = H * log(k) + c
                log_k = np.log(k_values)
                log_rs = np.log(rs_values)

                # 计算回归系数
                reg = np.polyfit(log_k, log_rs, 1)
                hurst = reg[0]

                # 赫斯特指数应该在0到1之间
                if hurst < 0 or hurst > 1:
                    return np.nan

                return hurst

            except Exception as e:
                return np.nan

        # 使用对数收益率计算Hurst指数
        log_returns = np.log(df["close"] / df["close"].shift(1))
        df["hurst_exponent"] = log_returns.rolling(
            window=120,
            min_periods=60  # 要求至少60个数据点
        ).apply(calculate_hurst)

        # 2. 偏度 (20日)
        df["skewness"] = returns.rolling(window=20).skew()

        # 3. 峰度 (20日)
        df["kurtosis"] = returns.rolling(window=20).kurt()

        # 按日期升序排序
        df = df.sort_values("date")

        # 使用 forward fill 填充技术指标的 NaN 值
        # 技术指标列（不包括基础价格数据）
        indicator_columns = [
            "momentum_1m", "momentum_3m", "momentum_6m",
            "volume_ma20", "volume_momentum",
            "historical_volatility", "volatility_regime", "volatility_z_score",
            "atr", "atr_ratio",
            "hurst_exponent", "skewness", "kurtosis"
        ]

        for col in indicator_columns:
            if col in df.columns:
                # 先用 forward fill 填充中间的 NaN
                df[col] = df[col].ffill()
                # 再用 backward fill 填充开头的 NaN（使用后续有效值）
                df[col] = df[col].bfill()
                # 如果仍有 NaN（如数据不足），用 0 填充
                df[col] = df[col].fillna(0)

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(
            f"Successfully fetched price history data ({len(df)} records)")

        # 检查并报告NaN值（仅检查基础数据列，技术指标已填充）
        base_columns = ["open", "close", "high", "low", "volume"]
        nan_columns = df[base_columns].isna().sum()
        if nan_columns.any():
            logger.warning(
                "\nWarning: The following base columns contain NaN values:")
            for col, nan_count in nan_columns[nan_columns > 0].items():
                logger.warning(f"- {col}: {nan_count} records")

        return df

    except Exception as e:
        _log_data_source_failure("Price history source", e)
        return pd.DataFrame()


def prices_to_df(prices):
    """Convert price data to DataFrame with standardized column names"""
    try:
        df = pd.DataFrame(prices)

        # 标准化列名映射
        column_mapping = {
            '收盘': 'close',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_percent',
            '涨跌额': 'change_amount',
            '换手率': 'turnover_rate'
        }

        # 重命名列
        for cn, en in column_mapping.items():
            if cn in df.columns:
                df[en] = df[cn]

        # 确保必要的列存在
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0  # 使用0填充缺失的必要列

        return df
    except Exception as e:
        logger.error(f"Error converting price data: {str(e)}")
        # 返回一个包含必要列的空DataFrame
        return pd.DataFrame(columns=['close', 'open', 'high', 'low', 'volume'])


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """获取股票价格数据

    Args:
        ticker: 股票代码
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD

    Returns:
        包含价格数据的DataFrame
    """
    return get_price_history(ticker, start_date, end_date)


def get_industry(symbol: str) -> str:
    """获取股票所属行业

    Args:
        symbol: 股票代码

    Returns:
        行业名称，如果获取失败返回空字符串
    """
    industry_cache_file = "src/data/industry_cache.json"

    # 尝试从缓存获取
    try:
        if os.path.exists(industry_cache_file):
            with open(industry_cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                if symbol in cache:
                    cached_industry = cache[symbol].get("industry", "")
                    logger.info(f"使用缓存的行业信息: {cached_industry}")
                    return cached_industry
    except Exception as e:
        logger.warning(f"读取行业缓存失败: {e}")

    # 备选方案1: 通过研报数据获取行业
    try:
        import akshare as ak
        rating_df = ak.stock_research_report_em(symbol=symbol)
        if rating_df is not None and not rating_df.empty and '行业' in rating_df.columns:
            industry = str(rating_df.iloc[0].get('行业', ''))
            if industry and industry not in ['nan', 'None', '']:
                logger.info(f"通过研报获取行业: {industry}")
                _cache_industry(symbol, industry, industry_cache_file)
                return industry
    except Exception as e:
        logger.debug(f"研报获取行业失败: {e}")

    # 备选方案2: 通过 stock_info_a_code_name 获取
    try:
        stock_info = ak.stock_info_a_code_name()
        if stock_info is not None and not stock_info.empty:
            match = stock_info[stock_info['code'] == symbol]
            if not match.empty:
                name = str(match.iloc[0].get('name', ''))
                if name:
                    # 可以尝试通过名称匹配行业，但这不太可靠
                    logger.info(f"找到股票名称: {name}")
    except Exception as e:
        logger.debug(f"stock_info_a_code_name 获取失败: {e}")

    logger.warning(f"无法获取股票 {symbol} 的行业信息")
    return ""


def _cache_industry(symbol: str, industry: str, cache_file: str) -> None:
    """缓存行业信息"""
    try:
        cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        cache[symbol] = {
            "industry": industry,
            "updated": datetime.now().isoformat()
        }
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"行业信息已缓存: {symbol} -> {industry}")
    except Exception as e:
        logger.warning(f"缓存行业信息失败: {e}")
