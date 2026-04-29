function normalizeConfidence(conf) {
    if (conf === undefined || conf === null || conf === '') return '-';
    const num = typeof conf === 'number' ? conf : parseFloat(conf);
    if (isNaN(num)) return '-';
    if (num > 1) return Math.round(num) + '%';
    return Math.round(num * 100) + '%';
}

const AGENT_DESCRIPTIONS = {
    'market_data_agent': `【当前任务】收集股票的基础数据，为后续分析提供原材料

【执行逻辑】从数据源获取价格、财务指标、行业分类。检查数据完整性，不足时标记。

【结果输出】最新价、市值、行业、数据区间覆盖情况`,

    'technical_analyst_agent': `【当前任务】基于价格和成交量判断短期走势

【执行逻辑】5策略并行：趋势跟踪(MA/EMA)、均值回归(RSI/KDJ)、动量(MACD)、波动率(ATR)、统计套利。多策略加权输出综合信号。

【结果输出】各策略信号列表、综合信号(看多/中性/看空)、置信度`,

    'fundamentals_agent': `【当前任务】评估公司盈利能力和财务健康状况

【执行逻辑】三维评估：盈利能力(ROE大于15%看涨、小于5%看跌)、成长性(营收增长大于10%看涨、小于0%看跌)、财务健康(流动比率大于2低风险、小于1高风险)。

【结果输出】盈利能力信号、成长性信号、财务健康信号、综合信号`,

    'sentiment_agent': `【当前任务】量化市场情绪和资金动向

【执行逻辑】四维度加权：新闻情绪(权重25%)、股吧热度(权重25%)、量化指标(权重25%)、北向资金(权重25%)。综合分数正向看涨、负向看跌。

【结果输出】综合情绪分数、情绪信号(看多/中性/看空)、置信度`,

    'valuation_agent': `【当前任务】判断当前价格是否偏离合理价值

【执行逻辑】四方法综合：DCF现金流折现、所有者收益(巴菲特公式)、PE/PB分位点、清算价值。计算加权合理价与当前价对比。

【结果输出】当前价对比合理价、折扣率、各方法信号、综合信号`,

    'industry_cycle_agent': `【当前任务】判断所处行业周期阶段

【执行逻辑】识别行业类型(周期/成长/防御)。分析行业景气指标。判断阶段(复苏/繁荣/衰退/萧条)。不同阶段对应不同配置权重。

【结果输出】行业类型、当前阶段、配置权重因子、行业信号`,

    'institutional_agent': `【当前任务】追踪大资金动向

【执行逻辑】两维度判断：基金持仓变化(增持大于5%看涨、减持大于5%看跌)、北向资金净流入(流入看涨、流出看跌)。

【结果输出】基金动向信号、北向资金信号、综合信号`,

    'expectation_diff_agent': `【当前任务】寻找市场预期差机会

【执行逻辑】两维度比较：业绩预告(预增表示实际可能超预期)、研报评级与实际表现差。

【结果输出】业绩预告信号、研报评级信号、预期差信号`,

    'macro_news_agent': `【当前任务】收集并整理宏观新闻

【执行逻辑】获取近期相关新闻。提取关键政策和经济数据。汇总形成结构化摘要。

【结果输出】新闻数量、关键政策要点、经济数据摘要`,

    'macro_analyst_agent': `【当前任务】评估宏观环境对个股的影响

【执行逻辑】三层推导：宏观环境评估(有利/不利/中性)、对个股影响路径(行业传导到基本面)、综合得出对股价影响(正面/负面/中性)。

【结果输出】宏观环境、对个股影响、关键因素列表、决策逻辑、具体操作建议`,

    'researcher_bull_agent': `【当前任务】汇总多方信号，形成看多逻辑

【执行逻辑】统计9个基础Agent中看涨/中性/看空数量。调用LLM综合分析。生成看多论点和置信度。

【结果输出】看多论点列表、置信度、看多信号`,

    'researcher_bear_agent': `【当前任务】汇总空方信号，形成看空逻辑

【执行逻辑】统计9个基础Agent中看涨/中性/看空数量。调用LLM综合分析。生成风险论点和置信度。

【结果输出】风险点列表、置信度、看空信号`,

    'debate_room_agent': `【当前任务】多空辩论，形成最终倾向

【执行逻辑】比较看多/看空研究员论点数量和置信度。LLM分析判断。一致时增强置信度，分歧时降低置信度。

【结果输出】辩论结论、调整后置信度、多空信号`,

    'risk_management_agent': `【当前任务】评估风险约束，确定仓位上限

【执行逻辑】综合评分：大盘风险(权重40%)加个股风险(权重40%)加相对风险(权重20%)。结合波动率、VaR、最大回撤计算动态风险评分。高于阈值减仓。

【结果输出】风险评分、风险指标详情、最大持仓规模、仓位建议`,

    'portfolio_management_agent': `【当前任务】综合所有输入，给出最终交易决策

【执行逻辑】综合辩论结论(权重50%)加风险评估(权重30%)加估值水平(权重20%)。LLM最终推理。输出动作、数量、置信度。

【结果输出】最终动作(买入/卖出/持有)、建议数量、置信度、决策理由`
};

const AGENT_DISPLAY_NAMES = {
    'market_data_agent': '市场数据',
    'technical_analyst_agent': '技术分析',
    'fundamentals_agent': '基本面',
    'sentiment_agent': '情绪分析',
    'valuation_agent': '估值分析',
    'industry_cycle_agent': '行业周期',
    'institutional_agent': '机构持仓',
    'expectation_diff_agent': '预期差',
    'macro_news_agent': '宏观新闻',
    'macro_analyst_agent': '宏观分析',
    'researcher_bull_agent': '多头研究',
    'researcher_bear_agent': '空头研究',
    'debate_room_agent': '辩论室',
    'risk_management_agent': '风险管理',
    'portfolio_management_agent': '最终决策'
};

const WORKFLOW_STAGES = [
    {
        key: 'level0',
        title: 'Level 0 · 数据收集',
        subtitle: '先准备价格、财务、行业与市场基础数据',
        agents: ['market_data_agent']
    },
    {
        key: 'level1',
        title: 'Level 1 · 九维并行分析',
        subtitle: '九个基础分析模块并行执行',
        agents: [
            'technical_analyst_agent',
            'fundamentals_agent',
            'sentiment_agent',
            'valuation_agent',
            'industry_cycle_agent',
            'institutional_agent',
            'expectation_diff_agent',
            'macro_news_agent',
            'macro_analyst_agent'
        ]
    },
    {
        key: 'level2',
        title: 'Level 2 · 多空研究',
        subtitle: '看多与看空研究员汇总九维观点',
        agents: ['researcher_bull_agent', 'researcher_bear_agent']
    },
    {
        key: 'level3',
        title: 'Level 3 · 辩论与风控',
        subtitle: '先形成辩论结论，再做风险约束',
        agents: ['debate_room_agent', 'risk_management_agent']
    },
    {
        key: 'level4',
        title: 'Level 4 · 最终决策',
        subtitle: '给出最终交易动作、数量与置信度',
        agents: ['portfolio_management_agent']
    }
];

const AGENT_ALIASES = {
    'technical_analyst': 'technical_analyst_agent',
    'fundamentals': 'fundamentals_agent',
    'sentiment': 'sentiment_agent',
    'valuation': 'valuation_agent',
    'industry_cycle': 'industry_cycle_agent',
    'institutional': 'institutional_agent',
    'expectation_diff': 'expectation_diff_agent',
    'macro_analyst': 'macro_analyst_agent',
    'researcher_bull': 'researcher_bull_agent',
    'researcher_bear': 'researcher_bear_agent',
    'debate_room': 'debate_room_agent',
    'risk_management': 'risk_management_agent',
    'portfolio_management': 'portfolio_management_agent',
    'market_data': 'market_data_agent',
    '技术分析师': 'technical_analyst_agent',
    '基本面分析师': 'fundamentals_agent',
    '情绪分析师': 'sentiment_agent',
    '估值Agent': 'valuation_agent',
    '行业周期分析师': 'industry_cycle_agent',
    '机构持仓分析师': 'institutional_agent',
    '预期差分析师': 'expectation_diff_agent',
    '宏观新闻Agent': 'macro_news_agent',
    '宏观分析师': 'macro_analyst_agent',
    '看多研究员': 'researcher_bull_agent',
    '看空研究员': 'researcher_bear_agent',
    '辩论室': 'debate_room_agent',
    '风险管理师': 'risk_management_agent',
    '投资组合管理': 'portfolio_management_agent',
    '市场数据Agent': 'market_data_agent',
    '技术分析': 'technical_analyst_agent',
    '基本面': 'fundamentals_agent',
    '情绪分析': 'sentiment_agent',
    '估值分析': 'valuation_agent',
    '行业周期': 'industry_cycle_agent',
    '机构持仓': 'institutional_agent',
    '预期差': 'expectation_diff_agent',
    '宏观新闻': 'macro_news_agent',
    '宏观分析': 'macro_analyst_agent',
    '多头研究': 'researcher_bull_agent',
    '空头研究': 'researcher_bear_agent',
    '辩论室': 'debate_room_agent',
    '风险管理': 'risk_management_agent',
    '组合管理': 'portfolio_management_agent',
    '最终决策': 'portfolio_management_agent',
    '投资组合决策': 'portfolio_management_agent',
    '市场数据': 'market_data_agent'
};

function normalizeAgentName(agentName) {
    if (!agentName) return agentName;
    if (AGENT_DISPLAY_NAMES[agentName]) return agentName;
    if (AGENT_ALIASES[agentName]) return AGENT_ALIASES[agentName];

    const cleaned = agentName.split(':')[0].trim();
    if (AGENT_DISPLAY_NAMES[cleaned]) return cleaned;
    if (AGENT_ALIASES[cleaned]) return AGENT_ALIASES[cleaned];

    return agentName;
}

const Components = {
    createAgentCard(agentName, state = {}) {
        const displayName = AGENT_DISPLAY_NAMES[agentName] || agentName;
        const status = state.status || 'pending';
        const signal = state.signal || null;
        const confidence = state.confidence || 0;
        const message = state.message || '';
        const isActive = state.isActive ? 'active' : '';

        return `
            <div class="agent-card status-${status} ${isActive}" data-agent="${agentName}">
                <div class="agent-header">
                    <span class="agent-name">${displayName}</span>
                    <span class="agent-status-badge ${status}">${this.getStatusText(status)}</span>
                </div>
                ${signal ? `<div class="agent-signal ${signal}">信号: ${this.getSignalText(signal)}</div>` : ''}
                ${confidence > 0 ? `
                    <div class="agent-confidence">
                        置信度: ${normalizeConfidence(confidence)}
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${Math.min(100, Math.max(0, confidence > 1 ? confidence : confidence * 100))}%"></div>
                    </div>
                ` : ''}
                ${message ? `<div class="agent-message">${this.escapeHtml(message)}</div>` : ''}
            </div>
        `;
    },

    getStatusText(status) {
        const statusMap = {
            'pending': '等待',
            'running': '执行中',
            'completed': '完成',
            'failed': '失败'
        };
        return statusMap[status] || status;
    },

    getSignalText(signal) {
        const signalMap = {
            'bullish': '看多',
            'bearish': '看空',
            'neutral': '中性',
            'positive': '正向',
            'negative': '负向',
            'buy': '买入',
            'sell': '卖出',
            'hold': '持有',
            'reduce': '减仓',
            'unavailable': '不可用',
            'error': '异常'
        };
        return signalMap[signal] || signal;
    },

    createLogEntry(log) {
        const timestamp = log.timestamp ? this.formatTime(log.timestamp) : '';
        const normalizedAgent = normalizeAgentName(log.agent);
        const agent = normalizedAgent ? (AGENT_DISPLAY_NAMES[normalizedAgent] || normalizedAgent) : '';
        const level = log.level || 'info';
        const message = log.message || '';

        return `
            <div class="log-entry">
                ${timestamp ? `<span class="log-timestamp">[${timestamp}]</span>` : ''}
                ${agent ? `<span class="log-agent">${agent}:</span>` : ''}
                <span class="log-message ${level}">${this.escapeHtml(message)}</span>
            </div>
        `;
    },

    createAgentGrid(agents) {
        const html = Object.entries(agents).map(([name, state]) => 
            this.createAgentCard(name, state)
        ).join('');
        return html;
    },

    createAgentWorkflow(agents) {
        return `
            <div class="workflow-map">
                ${WORKFLOW_STAGES.map((stage, index) => {
                    const cards = stage.agents.map((name) => this.createAgentCard(name, agents[name] || {})).join('');
                    const layoutClass = stage.agents.length === 1 ? 'single' : stage.agents.length <= 2 ? 'pair' : 'matrix';
                    return `
                        <section class="workflow-stage ${layoutClass}" data-stage="${stage.key}">
                            <div class="workflow-stage-header">
                                <div>
                                    <div class="workflow-stage-title">${stage.title}</div>
                                    <div class="workflow-stage-subtitle">${stage.subtitle}</div>
                                </div>
                                <div class="workflow-stage-progress">${this.getStageStatusText(stage, agents)}</div>
                            </div>
                            <div class="workflow-stage-body ${layoutClass}">
                                ${cards}
                            </div>
                            ${index < WORKFLOW_STAGES.length - 1 ? '<div class="workflow-stage-arrow"><span></span></div>' : ''}
                        </section>
                    `;
                }).join('')}
            </div>
        `;
    },

    getStageStatusText(stage, agents) {
        const states = stage.agents.map((name) => agents[name]?.status || 'pending');
        if (states.some((status) => status === 'running')) return '执行中';
        if (states.every((status) => status === 'completed')) return '已完成';
        if (states.some((status) => status === 'failed')) return '异常';
        return '等待中';
    },

    createResultPanel(result) {
        if (!result) return '<div class="log-empty">暂无结果</div>';

        const action = result.action || 'unknown';
        const confidence = result.confidence || 0;
        const quantity = result.quantity || 0;
        const reasoning = result.reasoning || '';
        const agentSignals = Array.isArray(result.agent_signals) ? result.agent_signals : [];
        const reasoningLines = String(reasoning)
            .split('\n')
            .map((line) => line.trim())
            .filter(Boolean);
        const headline = reasoningLines[0] || '系统已生成最终决策';
        const detailLines = reasoningLines.slice(1, 7);
        const bullishSignals = agentSignals.filter((s) => ['bullish', 'buy', 'positive'].includes(String(s.signal || '').toLowerCase()));
        const bearishSignals = agentSignals.filter((s) => ['bearish', 'sell', 'negative', 'reduce'].includes(String(s.signal || '').toLowerCase()));
        const neutralSignals = agentSignals.filter((s) => ['neutral', 'hold'].includes(String(s.signal || '').toLowerCase()));
        const readableSummary = this.buildDecisionNarrative(action, confidence, quantity, bullishSignals, bearishSignals, neutralSignals);

        return `
            <div class="result-summary">
                <div class="result-item">
                    <div class="result-label">行动</div>
                    <div class="result-value action-${action}">${this.getActionText(action)}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">置信度</div>
                    <div class="result-value">${normalizeConfidence(confidence)}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">建议数量</div>
                    <div class="result-value">${quantity}股</div>
                </div>
            </div>
            <div class="decision-highlight action-${action}">
                <div class="decision-highlight-label">结论摘要</div>
                <div class="decision-highlight-text">${this.escapeHtml(headline)}</div>
            </div>
            <div class="result-reasoning">
                <h4>决策逻辑解读</h4>
                <div class="decision-points">
                    ${readableSummary.map((line) => `<div class="decision-point">${this.escapeHtml(line)}</div>`).join('')}
                </div>
            </div>
            ${detailLines.length ? `
                <div class="result-reasoning">
                    <h4>关键依据</h4>
                    <div class="decision-points">
                        ${detailLines.map((line) => `<div class="decision-point">${this.escapeHtml(line)}</div>`).join('')}
                    </div>
                </div>
            ` : ''}
            ${reasoning ? `
                <div class="result-reasoning">
                    <h4>完整决策说明</h4>
                    <p>${this.escapeHtml(reasoning)}</p>
                </div>
            ` : ''}
            ${agentSignals.length ? `
                <div class="result-reasoning">
                    <h4>各模块决策输入</h4>
                    <div class="signal-grid">
                        ${agentSignals.map((signal) => this.createSignalCard(signal)).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    },

    buildDecisionNarrative(action, confidence, quantity, bullishSignals, bearishSignals, neutralSignals) {
        const lines = [];
        const actionText = this.getActionText(action);
        lines.push(`最终建议为${actionText}，建议数量 ${quantity} 股，系统整体把握度约 ${normalizeConfidence(confidence)}。`);

        if (bullishSignals.length) {
            lines.push(`偏正向的模块有 ${bullishSignals.map((s) => s.agent_name).join('、')}。`);
        }
        if (bearishSignals.length) {
            lines.push(`偏负向的模块有 ${bearishSignals.map((s) => s.agent_name).join('、')}。`);
        }
        if (neutralSignals.length) {
            lines.push(`保持中性或观望的模块有 ${neutralSignals.map((s) => s.agent_name).join('、')}。`);
        }

        if (action === 'hold') {
            lines.push('这通常意味着当前多空信号尚未形成足够一致的优势，或者风险管理要求暂时观望。');
        } else if (action === 'buy') {
            lines.push('这说明正向信号在关键模块中占优，并且当前风险约束允许建立或增加仓位。');
        } else if (action === 'sell') {
            lines.push('这说明负向信号或风险约束占优，系统更倾向于降低敞口或退出。');
        }

        return lines;
    },

createAgentDetail(agentName, state = {}, logs = []) {
        const displayName = AGENT_DISPLAY_NAMES[agentName] || agentName;
        const status = state.status || 'pending';
        const signal = state.signal || null;
        const confidence = state.confidence || 0;
        const details = state.details || {};
        const signalText = signal ? this.getSignalText(signal) : (status === 'completed' ? '无信号' : '等待中');
        const confidenceText = normalizeConfidence(confidence);
        const summary = details.summary || '';
        const resultData = details.result || {};

        const resultText = this.buildResultText(agentName, resultData, signal);

        const recentLogs = logs.slice(-20);
        const logHtml = recentLogs.length
            ? recentLogs.map((log, i) => `<div class="log-line"><span class="log-num">${i+1}</span><span class="log-content">${this.escapeHtml(log.message || '')}</span></div>`).join('')
            : '<div class="empty">暂无日志</div>';

        const description = AGENT_DESCRIPTIONS[agentName] || '';

        return `
            <div class="detail-shell">
                <div class="detail-header">
                    <div class="detail-title">
                        <span class="detail-name">${displayName}</span>
                        <span class="detail-badge ${status}">${status === 'completed' ? '✓' : status === 'running' ? '⟳' : '○'}</span>
                    </div>
                    <div class="detail-meta">
                        <span class="meta-signal ${signal || ''}">${signalText}</span>
                        <span class="meta-confidence">置信度 ${confidenceText}</span>
                    </div>
                </div>
                ${description ? `<div class="detail-description">
                    <div class="detail-section-label">模块说明</div>
                    <div class="detail-description-text">${description}</div>
                </div>` : ''}
                ${resultText ? `<div class="detail-result">${resultText}</div>` : ''}
                ${this.buildDetailData(agentName, resultData)}
                <div class="detail-logs">
                    <div class="logs-title">执行过程</div>
                    <div class="logs-box">${logHtml}</div>
                </div>
            </div>
        `;
    },

    buildResultText(agentName, data, signal) {
        if (!data || Object.keys(data).length === 0) return '';
        
        const d = data;
        if (agentName === 'technical_analyst_agent') {
            const signals = d.strategy_signals || {};
            const strategyNames = {
                'trend_following': '趋势跟踪',
                'mean_reversion': '均值回归',
                'momentum': '动量',
                'volatility': '波动率',
                'statistical_arbitrage': '统计套利'
            };
            const parts = Object.keys(signals).map(key => {
                const signal = signals[key]?.signal || '-';
                const name = strategyNames[key] || key;
                return `${name}[${this.getSignalText(signal)}]`;
            });
            return parts.length > 0 ? `技术面：${parts.join('，')}` : '技术分析暂无数据';
        }
        if (agentName === 'fundamentals_agent') {
            const r = d.reasoning || {};
            const hasFallback = r.fallback ? true : false;
            const profit = d.profitability_signal?.signal || r.profitability_signal?.signal || r.fallback?.signal || '-';
            const growth = d.growth_signal?.signal || r.growth_signal?.signal || r.fallback?.signal || '-';
            const health = d.financial_health_signal?.signal || r.financial_health_signal?.signal || r.fallback?.signal || '-';
            const profitText = this.getSignalText(profit);
            const growthText = this.getSignalText(growth);
            const healthText = this.getSignalText(health);
            const status = hasFallback ? '(数据不足)' : '';
            return `基本面${status}：盈利能力[${profitText}]，成长性[${growthText}]，财务健康[${healthText}]`;
        }
        if (agentName === 'sentiment_agent') {
            const signalText = this.getSignalText(d.signal);
            const confidence = d.confidence || '-';
            const score = d.combined_score !== undefined ? d.combined_score.toFixed(3) : '-';
            return `情绪分析：${signalText}（置信度${confidence}），综合分数${score}`;
        }
        if (agentName === 'valuation_agent') {
            const signalText = this.getSignalText(d.signal);
            const confNum = typeof d.confidence === 'number' ? d.confidence : parseFloat(d.confidence);
            const confPct = !isNaN(confNum) ? Math.round(confNum * 100) + '%' : (d.confidence || '-');
            const price = d.current_price != null ? Number(d.current_price).toFixed(2) + '元' : '-';
            const fair = d.fair_value != null ? Number(d.fair_value).toFixed(2) + '元' : '-';
            const discount = d.discount != null ? (Number(d.discount) * 100).toFixed(1) + '%' : '-';
            const methodNames = d.methods ? d.methods.map(m => m.name).join('、') : (d.reasoning ? Object.keys(d.reasoning).join('、') : '-');
            return `估值：${signalText}（置信度${confPct}），当前价${price}，合理价${fair}，折扣率${discount}，方法：${methodNames}`;
        }
        if (agentName === 'macro_analyst_agent') {
            const envMap = {'favorable': '有利', 'unfavorable': '不利', 'neutral': '中性', '有利': '有利', '不利': '不利', '中性': '中性'};
            const impactMap = {'positive': '正面', 'negative': '负面', 'neutral': '中性', '正面': '正面', '负面': '负面', '中性': '中性'};
            const env = envMap[d.macro_environment] || d.macro_environment || '-';
            const impact = impactMap[d.impact_on_stock] || d.impact_on_stock || '-';
            const decisionResult = d.decision_result || {};
            const action = decisionResult.action || '';
            const recommendation = decisionResult.recommendation || '';
            const summary = recommendation || action ? `，建议${recommendation || action}` : '';
            return `宏观分析：环境${env}，对股票${impact}，置信度${normalizeConfidence(d.confidence)}${summary}`;
        }
        if (agentName === 'macro_news_agent') {
            const newsResult = d.macro_news_analysis_result || d;
            const newsCount = newsResult.retrieved_news_count || newsResult.获取新闻数 || 0;
            const summary = newsResult.summary_content || newsResult.新闻摘要 || newsResult.summary || newsResult.最终输出 || newsResult.reasoning || '';
            if (summary) {
                return `宏观新闻：获取${newsCount}条，${summary.slice(0, 80)}...`;
            }
            return `宏观新闻：暂无数据`;
        }
        if (agentName === 'risk_management_agent') {
            const riskScore = d.风险评分 || d.risk_score || '-';
            const action = d.交易行动 || d.trading_action || 'hold';
            const maxPos = d.最大持仓规模 || d.max_position_size || '-';
            const marketRisk = d.风险指标?.大盘风险评分 || d.components?.market_risk || '-';
            const stockRisk = d.风险指标?.个股风险评分 || d.components?.stock_risk || '-';
            const volatility = d.风险指标?.波动率 || d.metrics?.stock_volatility || 0;
            const var95 = d.风险指标?.['95%风险价值(VaR)'] || d.metrics?.stock_var || 0;
            const actionMap = {buy: '买入', sell: '卖出', hold: '持有'};
            const actionText = actionMap[action] || action;
            return `风险评分：${riskScore}/10（大盘${marketRisk}分/个股${stockRisk}分），波动率${(volatility*100).toFixed(1)}%，VaR${(var95*100).toFixed(1)}%，建议：${actionText}，最大持仓：${typeof maxPos === 'number' ? Math.round(maxPos) : maxPos}`;
        }
        if (agentName === 'portfolio_management_agent') {
            const confNum = typeof d.confidence === 'number' ? d.confidence : parseFloat(d.confidence);
            const confPct = !isNaN(confNum) ? Math.round(confNum * 100) + '%' : '-';
            const reason = d.reasoning ? d.reasoning.slice(0, 80) + (d.reasoning.length > 80 ? '...' : '') : '';
            return `最终决策：${this.getActionText(d.action || 'hold')} ${d.quantity || 0}股，置信度${confPct}${reason ? ' — ' + reason : ''}`;
        }
        if (agentName === 'industry_cycle_agent') {
            const signalText = this.getSignalText(d.signal);
            const confNum = typeof d.confidence === 'number' ? d.confidence : parseFloat(d.confidence);
            const confPct = !isNaN(confNum) ? Math.round(confNum * 100) + '%' : (d.confidence || '-');
            const cycle = d.cycle_type_cn || d.cycle_type || '-';
            const phase = d.phase || '-';
            return `行业周期：${signalText}（置信度${confPct}），${cycle}，${phase}`;
        }
        if (agentName === 'institutional_agent') {
            const signalText = this.getSignalText(d.signal);
            const confNum = typeof d.confidence === 'number' ? d.confidence : parseFloat(d.confidence);
            const confPct = !isNaN(confNum) ? Math.round(confNum * 100) + '%' : (d.confidence || '-');
            const details = d.details || [];
            const preview = details.map(di => `${di.name}${di.signal_cn}`).join('，') || '-';
            return `机构持仓：${signalText}（置信度${confPct}），${preview}`;
        }
        if (agentName === 'expectation_diff_agent') {
            const signalText = this.getSignalText(d.signal);
            const forecast = d.earnings_forecast || d.业绩预告 || {};
            const rating = d.research_rating || d.研报评级 || {};
            const forecastText = this.getSignalText(forecast.signal);
            const ratingText = this.getSignalText(rating.signal);
            return `预期差分析：${signalText}，预告${forecastText}，评级${ratingText}`;
        }
        if (agentName === 'researcher_bull_agent') {
            const points = d.thesis_points || [];
            const preview = points.length > 0 ? points.slice(0, 2).join('；') : '无具体论点';
            return `看多研究：置信度${d.confidence ?? '-'}，论点${points.length}条：${preview}`;
        }
        if (agentName === 'researcher_bear_agent') {
            const points = d.risk_points || [];
            const preview = points.length > 0 ? points.slice(0, 2).join('；') : '无具体风险点';
            return `看空研究：置信度${d.confidence ?? '-'}，风险点${points.length}条：${preview}`;
        }
        if (agentName === 'debate_room_agent') {
            const signalText = this.getSignalText(d.signal);
            const summary = d.debate_summary || [];
            return `辩论结论：${signalText}（置信度${d.confidence ?? '-'}），论辩点${summary.length}条`;
        }
        if (agentName === 'market_data_agent') {
            const ticker = d.ticker || '-';
            const count = d.prices_count != null ? d.prices_count + '条' : '-';
            const industry = d.industry || '-';
            const price = d.latest_price != null ? d.latest_price.toFixed(2) + '元' : '-';
            const cap = d.market_cap ? (d.market_cap / 1e8).toFixed(1) + '亿' : '-';
            return `${ticker}：${count}价格记录，最新价${price}，市值${cap}，行业${industry}`;
        }
        
        return JSON.stringify(data).slice(0, 150);
    },

    buildDetailData(agentName, resultData) {
        if (!resultData || Object.keys(resultData).length === 0) return '';

        if (agentName === 'researcher_bull_agent') {
            const points = resultData.thesis_points || [];
            if (points.length === 0) return '';
            const items = points.map((p, i) => `<div class="detail-point"><span class="point-num">${i + 1}.</span>${this.escapeHtml(p)}</div>`).join('');
            return `<div class="detail-section"><div class="detail-section-title">看多论点 (${points.length}条)</div>${items}</div>`;
        }

        if (agentName === 'researcher_bear_agent') {
            const points = resultData.risk_points || [];
            if (points.length === 0) return '';
            const items = points.map((p, i) => `<div class="detail-point"><span class="point-num">${i + 1}.</span>${this.escapeHtml(p)}</div>`).join('');
            return `<div class="detail-section"><div class="detail-section-title">风险点 (${points.length}条)</div>${items}</div>`;
        }

        if (agentName === 'debate_room_agent') {
            const summary = resultData.debate_summary || [];
            const llmAnalysis = resultData.llm_analysis || '';
            let html = '';
            if (summary.length > 0) {
                const items = summary.map(s => `<div class="detail-point">${this.escapeHtml(s)}</div>`).join('');
                html += `<div class="detail-section"><div class="detail-section-title">辩论总结</div>${items}</div>`;
            }
            if (llmAnalysis) {
                html += `<div class="detail-section"><div class="detail-section-title">LLM分析</div><div class="detail-point">${this.escapeHtml(llmAnalysis)}</div></div>`;
            }
            return html;
        }

        if (agentName === 'risk_management_agent') {
            const riskMetrics = resultData.风险指标 || resultData.metrics || {};
            const debateAnalysis = resultData.辩论分析 || resultData.debate_analysis || {};
            const components = resultData.风险指标?.components || resultData.components || {};
            const actionMap = {buy: '买入', sell: '卖出', hold: '持有'};
            const action = resultData.交易行动 || resultData.trading_action || 'hold';
            const actionText = actionMap[action] || action;
            let html = '';

            html += `<div class="detail-section"><div class="detail-section-title">决策逻辑</div>`;
            html += `<div class="detail-point"><strong>风险评分：</strong>${resultData.风险评分 || resultData.risk_score || '-'}/10，动态阈值：${(resultData.动态阈值 || resultData.dynamic_threshold || 0).toFixed(1)}</div>`;
            html += `<div class="detail-point"><strong>评分构成：</strong>大盘风险${components.market_risk || riskMetrics.大盘风险评分 || 0}分 + 个股风险${components.stock_risk || riskMetrics.个股风险评分 || 0}分 + 相对风险${components.relative_risk || riskMetrics.相对风险评分 || 0}分</div>`;
            html += `<div class="detail-point"><strong>决策依据：</strong>${resultData.推理 || resultData.reasoning || '综合评估得出'}</div>`;
            html += `<div class="detail-point"><strong>最终建议：</strong>${actionText}（${action === 'hold' ? '风险评分高于阈值或辩论信号不明确' : action === 'buy' ? '风险可控，建议买入' : '风险过高，建议卖出'}）</div>`;
            html += `</div>`;

            html += `<div class="detail-section"><div class="detail-section-title">风险指标详情</div>`;
            html += `<div class="detail-point"><strong>波动率：</strong>${((riskMetrics.波动率 || riskMetrics.stock_volatility || 0) * 100).toFixed(2)}% （年化）</div>`;
            html += `<div class="detail-point"><strong>VaR(95%)：</strong>${((riskMetrics['95%风险价值(VaR)'] || riskMetrics.stock_var || 0) * 100).toFixed(2)}% （单日最大损失概率5%）</div>`;
            html += `<div class="detail-point"><strong>CVaR(95%)：</strong>${((riskMetrics['95%条件风险价值(CVaR)'] || riskMetrics.stock_cvar || 0) * 100).toFixed(2)}% （极端损失情况）</div>`;
            html += `<div class="detail-point"><strong>最大回撤：</strong>${((riskMetrics.最大回撤 || riskMetrics.stock_drawdown || 0) * 100).toFixed(2)}%</div>`;
            html += `</div>`;

            html += `<div class="detail-section"><div class="detail-section-title">大盘风险对比</div>`;
            html += `<div class="detail-point"><strong>大盘波动率：</strong>${((riskMetrics.大盘波动率 || riskMetrics.market_volatility || 0) * 100).toFixed(2)}%</div>`;
            html += `<div class="detail-point"><strong>大盘VaR：</strong>${((riskMetrics.大盘VaR || riskMetrics.market_var || 0) * 100).toFixed(2)}%</div>`;
            html += `<div class="detail-point"><strong>大盘回撤：</strong>${((riskMetrics.大盘回撤 || riskMetrics.market_drawdown || 0) * 100).toFixed(2)}%</div>`;
            html += `<div class="detail-point"><strong>大盘风险评分：</strong>${riskMetrics.大盘风险评分 || components.market_risk || '-'}/10</div>`;
            html += `</div>`;

            if (debateAnalysis.辩论信号) {
                html += `<div class="detail-section"><div class="detail-section-title">辩论信号影响</div>`;
                html += `<div class="detail-point"><strong>辩论信号：</strong>${debateAnalysis.辩论信号 || debateAnalysis.debate_signal || '-'}</div>`;
                html += `<div class="detail-point"><strong>多方置信度：</strong>${((debateAnalysis.多方置信度 || debateAnalysis.bull_confidence || 0) * 100).toFixed(0)}%</div>`;
                html += `<div class="detail-point"><strong>空方置信度：</strong>${((debateAnalysis.空方置信度 || debateAnalysis.bear_confidence || 0) * 100).toFixed(0)}%</div>`;
                html += `<div class="detail-point"><strong>辩论置信度：</strong>${((debateAnalysis.辩论置信度 || debateAnalysis.debate_confidence || 0) * 100).toFixed(0)}%</div>`;
                html += `</div>`;
            }

            const stress = riskMetrics.压力测试结果 || riskMetrics.stress_test_results || {};
            if (Object.keys(stress).length > 0) {
                html += `<div class="detail-section"><div class="detail-section-title">压力测试</div>`;
                const scenarios = {market_crash: '市场崩盘(-20%)', moderate_decline: '中度下跌(-10%)', slight_decline: '轻度下跌(-5%)'};
                for (const [key, label] of Object.entries(scenarios)) {
                    const result = stress[key] || {};
                    const loss = result.潜在损失 !== undefined ? Math.round(result.潜在损失 || 0) : '-';
                    const impact = result.组合影响 !== undefined ? ((result.组合影响 || 0) * 100).toFixed(1) + '%' : '-';
                    html += `<div class="detail-point"><strong>${label}：</strong>潜在损失${loss}元，组合影响${impact}</div>`;
                }
                html += `</div>`;
            }

            html += `<div class="detail-section"><div class="detail-section-title">仓位建议</div>`;
            html += `<div class="detail-point"><strong>最大持仓规模：</strong>${Math.round(resultData.最大持仓规模 || resultData.max_position_size || 0)}元</div>`;
            html += `<div class="detail-point"><strong>基础仓位：</strong>25%</div>`;
            const riskScore = resultData.风险评分 || resultData.risk_score || 0;
            const threshold = resultData.动态阈值 || resultData.dynamic_threshold || 5;
            let positionAdj = '基准';
            if (riskScore >= 8) positionAdj = '降至7.5%';
            else if (riskScore >= threshold) positionAdj = '降至12.5%';
            else if (riskScore >= 4) positionAdj = '降至18.75%';
            html += `<div class="detail-point"><strong>风险调整：</strong>${positionAdj}</div>`;
            html += `</div>`;

            return html;
        }

        if (agentName === 'valuation_agent') {
            const methods = resultData.methods || [];
            const reasoning = resultData.reasoning || {};
            let html = '';

            // 价格概览
            if (resultData.current_price != null || resultData.fair_value != null) {
                const cp = resultData.current_price != null ? Number(resultData.current_price).toFixed(2) + '元' : '-';
                const fv = resultData.fair_value != null ? Number(resultData.fair_value).toFixed(2) + '元' : '-';
                const disc = resultData.discount != null ? (Number(resultData.discount) * 100).toFixed(1) + '%' : '-';
                html += `<div class="detail-section">
                    <div class="detail-section-title">价格对比</div>
                    <div class="valuation-metrics">
                        <div class="val-metric"><span class="val-label">当前价</span><span class="val-value">${cp}</span></div>
                        <div class="val-metric"><span class="val-label">合理价</span><span class="val-value">${fv}</span></div>
                        <div class="val-metric"><span class="val-label">折扣率</span><span class="val-value">${disc}</span></div>
                    </div>
                </div>`;
            }

            // PE/PB 参考
            if (resultData.pe_ratio != null || resultData.pb_ratio != null) {
                const pe = resultData.pe_ratio != null ? resultData.pe_ratio.toFixed(1) : '-';
                const pb = resultData.pb_ratio != null ? resultData.pb_ratio.toFixed(1) : '-';
                html += `<div class="detail-section">
                    <div class="detail-section-title">估值指标</div>
                    <div class="valuation-metrics">
                        <div class="val-metric"><span class="val-label">PE</span><span class="val-value">${pe}</span></div>
                        <div class="val-metric"><span class="val-label">PB</span><span class="val-value">${pb}</span></div>
                        <div class="val-metric"><span class="val-label">股票类型</span><span class="val-value">${resultData.stock_type || '-'}</span></div>
                    </div>
                </div>`;
            }

            // 各估值方法详情
            if (methods.length > 0) {
                const methodItems = methods.map(m => {
                    const sigClass = `signal-tag ${m.signal || 'neutral'}`;
                    return `<div class="method-row">
                        <span class="method-name">${this.escapeHtml(m.name)}</span>
                        <span class="${sigClass}">${this.getSignalText(m.signal)}</span>
                        <span class="method-weight">权重 ${Math.round((m.weight || 0) * 100)}%</span>
                    </div>`;
                }).join('');
                html += `<div class="detail-section">
                    <div class="detail-section-title">估值方法详情</div>
                    <div class="method-list">${methodItems}</div>
                </div>`;
            }

            // 各方法详细说明（从reasoning中提取details）
            const detailEntries = [];
            Object.keys(reasoning).forEach(key => {
                const item = reasoning[key];
                if (item && item.details) {
                    const sigText = this.getSignalText(item.signal);
                    detailEntries.push(`<div class="detail-point"><span class="point-num sig-${item.signal || 'neutral'}">${sigText}</span>${this.escapeHtml(item.details)}</div>`);
                }
            });
            if (detailEntries.length > 0) {
                html += `<div class="detail-section">
                    <div class="detail-section-title">方法说明</div>
                    ${detailEntries.join('')}
                </div>`;
            }

            return html || '';
        }

        if (agentName === 'industry_cycle_agent') {
            const signalText = this.getSignalText(resultData.signal);
            const confPct = normalizeConfidence(resultData.confidence);
            const cycle = resultData.cycle_type_cn || resultData.cycle_type || '-';
            const phase = resultData.phase || '-';
            const logic = resultData.decision_logic || resultData.reason || '';
            const wf = resultData.weight_factor != null ? (resultData.weight_factor * 100).toFixed(0) + '%' : '-';

            return `<div class="detail-section">
                <div class="detail-section-title">周期判断</div>
                <div class="valuation-metrics">
                    <div class="val-metric"><span class="val-label">行业</span><span class="val-value">${resultData.industry || '-'}</span></div>
                    <div class="val-metric"><span class="val-label">周期类型</span><span class="val-value">${cycle}</span></div>
                    <div class="val-metric"><span class="val-label">当前阶段</span><span class="val-value">${phase}</span></div>
                    <div class="val-metric"><span class="val-label">配置权重</span><span class="val-value">${wf}</span></div>
                </div>
            </div>
            <div class="detail-section">
                <div class="detail-section-title">决策结果</div>
                <div class="detail-point"><span class="point-num sig-${resultData.signal || 'neutral'}">${signalText}</span>置信度 ${confPct}</div>
            </div>
            ${logic ? `<div class="detail-section">
                <div class="detail-section-title">决策逻辑</div>
                <div class="detail-point">${this.escapeHtml(logic)}</div>
            </div>` : ''}`;
        }

        if (agentName === 'institutional_agent') {
            const details = resultData.details || [];
            let html = '';

            if (resultData.signal) {
                const signalText = this.getSignalText(resultData.signal);
                const confPct = normalizeConfidence(resultData.confidence);
                html += `<div class="detail-section">
                    <div class="detail-section-title">综合判断</div>
                    <div class="detail-point"><span class="point-num sig-${resultData.signal}">${signalText}</span>置信度 ${confPct}</div>
                </div>`;
            }

            if (details.length > 0) {
                const items = details.map(d => {
                    const sig = d.signal || 'neutral';
                    return `<div class="method-row">
                        <span class="method-name">${this.escapeHtml(d.name)}</span>
                        <span class="signal-tag ${sig}">${d.signal_cn || this.getSignalText(sig)}</span>
                        <span class="method-weight">${this.escapeHtml(d.reason || '')}</span>
                    </div>`;
                }).join('');
                html += `<div class="detail-section">
                    <div class="detail-section-title">各维度详情</div>
                    <div class="method-list">${items}</div>
                </div>`;
            }

            if (resultData.decision_logic) {
                html += `<div class="detail-section">
                    <div class="detail-section-title">决策逻辑</div>
                    <div class="detail-point">${this.escapeHtml(resultData.decision_logic)}</div>
                </div>`;
            }

            return html || '';
        }

        if (agentName === 'market_data_agent') {
            let html = '';
            const ticker = resultData.ticker || '-';
            const startDate = resultData.start_date || '-';
            const endDate = resultData.end_date || '-';
            const price = resultData.latest_price != null ? Number(resultData.latest_price).toFixed(2) + '元' : '-';
            const cap = resultData.market_cap ? (Number(resultData.market_cap) / 1e8).toFixed(1) + '亿' : '-';
            const count = resultData.prices_count != null ? resultData.prices_count + '条' : '-';
            const industry = resultData.industry || '-';
            const hasMetrics = resultData.has_financial_metrics ? '✓' : '✗';
            const hasStatements = resultData.has_financial_statements ? '✓' : '✗';

            html += `<div class="detail-section">
                <div class="detail-section-title">基本信息</div>
                <div class="valuation-metrics">
                    <div class="val-metric"><span class="val-label">股票代码</span><span class="val-value">${ticker}</span></div>
                    <div class="val-metric"><span class="val-label">最新价</span><span class="val-value">${price}</span></div>
                    <div class="val-metric"><span class="val-label">市值</span><span class="val-value">${cap}</span></div>
                    <div class="val-metric"><span class="val-label">行业</span><span class="val-value">${industry}</span></div>
                </div>
            </div>`;

            html += `<div class="detail-section">
                <div class="detail-section-title">数据覆盖</div>
                <div class="valuation-metrics">
                    <div class="val-metric"><span class="val-label">数据区间</span><span class="val-value" style="font-size:12px">${startDate} ~ ${endDate}</span></div>
                    <div class="val-metric"><span class="val-label">价格记录</span><span class="val-value">${count}</span></div>
                    <div class="val-metric"><span class="val-label">财务指标</span><span class="val-value">${hasMetrics}</span></div>
                    <div class="val-metric"><span class="val-label">财务报表</span><span class="val-value">${hasStatements}</span></div>
                </div>
            </div>`;

            return html;
        }

        if (agentName === 'sentiment_agent') {
            const components = resultData.components || resultData.reasoning?.components || {};
            const weights = resultData.weights || {};
            const reasoning = resultData.reasoning || {};
            let html = '';

            html += `<div class="detail-section"><div class="detail-section-title">决策逻辑</div>`;
            html += `<div class="detail-point"><strong>综合信号：</strong>${this.getSignalText(resultData.signal)}</div>`;
            html += `<div class="detail-point"><strong>综合分数：</strong>${(resultData.combined_score || resultData.combined_score || 0).toFixed(3)}</div>`;
            if (reasoning.decision_logic) {
                html += `<div class="detail-point">${this.escapeHtml(reasoning.decision_logic)}</div>`;
            }
            html += `</div>`;

            const newsComp = components.news || {};
            const gubaComp = components.guba || {};
            const quantComp = components.quant || {};
            const northComp = components.north_money || {};

            html += `<div class="detail-section"><div class="detail-section-title">各情绪维度详情</div>`;
            html += `<div class="detail-point"><strong>新闻情绪：</strong>${(newsComp.score || 0).toFixed(2)}分，权重${(weights.news || 0.35) * 100}%，贡献${((newsComp.score || 0) * (weights.news || 0.35)).toFixed(3)}</div>`;
            html += `<div class="detail-point"><strong>股吧情绪：</strong>${(gubaComp.adjusted_score || gubaComp.score || 0).toFixed(2)}分，权重${(weights.guba || 0.15) * 100}%，帖子数${gubaComp.post_count || 0}</div>`;
            html += `<div class="detail-point"><strong>量化情绪：</strong>${(quantComp.score || 0).toFixed(2)}分，权重${(weights.quant || 0.35) * 100}%</div>`;
            const northSignal = northComp.signal || 'neutral';
            const northSignalText = {bullish: '净流入', bearish: '净流出', neutral: '持平'}[northSignal] || '未知';
            html += `<div class="detail-point"><strong>北向资金：</strong>${northSignalText}，权重${(weights.north_money || 0.15) * 100}%</div>`;
            html += `</div>`;

            return html;
        }

        if (agentName === 'fundamentals_agent') {
            const r = resultData.reasoning || {};
            const details = resultData.details || {};
            let html = '';

            html += `<div class="detail-section"><div class="detail-section-title">决策逻辑</div>`;
            html += `<div class="detail-point"><strong>综合信号：</strong>${this.getSignalText(resultData.signal)}</div>`;
            const conf = typeof resultData.confidence === 'number' ? resultData.confidence : parseFloat(resultData.confidence || 0);
            html += `<div class="detail-point"><strong>置信度：</strong>${normalizeConfidence(conf)}</div>`;
            if (r.decision_logic) {
                html += `<div class="detail-point">${this.escapeHtml(r.decision_logic)}</div>`;
            }
            html += `</div>`;

            html += `<div class="detail-section"><div class="detail-section-title">基本面三维度</div>`;
            const profit = resultData.profitability_signal?.signal || r.profitability_signal?.signal || '-';
            const profitDetails = resultData.profitability_signal?.details || r.profitability_signal?.details || '';
            html += `<div class="detail-point"><strong>盈利能力：</strong>${this.getSignalText(profit)} ${profitDetails ? '(' + profitDetails + ')' : ''}</div>`;

            const growth = resultData.growth_signal?.signal || r.growth_signal?.signal || '-';
            const growthDetails = resultData.growth_signal?.details || r.growth_signal?.details || '';
            html += `<div class="detail-point"><strong>成长性：</strong>${this.getSignalText(growth)} ${growthDetails ? '(' + growthDetails + ')' : ''}</div>`;

            const health = resultData.financial_health_signal?.signal || r.financial_health_signal?.signal || '-';
            const healthDetails = resultData.financial_health_signal?.details || r.financial_health_signal?.details || '';
            html += `<div class="detail-point"><strong>财务健康：</strong>${this.getSignalText(health)} ${healthDetails ? '(' + healthDetails + ')' : ''}</div>`;
            html += `</div>`;

            if (details.price_ratios) {
                html += `<div class="detail-section"><div class="detail-section-title">估值指标</div>`;
                html += `<div class="detail-point">${details.price_ratios.details || ''}</div>`;
                html += `</div>`;
            }

            return html;
        }

        if (agentName === 'technical_analyst_agent') {
            const signals = resultData.strategy_signals || {};
            const prices = resultData.prices || [];
            let html = '';

            html += `<div class="detail-section"><div class="detail-section-title">决策逻辑</div>`;
            html += `<div class="detail-point"><strong>综合信号：</strong>${this.getSignalText(resultData.signal)}</div>`;
            const conf = typeof resultData.confidence === 'number' ? resultData.confidence : parseFloat(resultData.confidence || 0);
            html += `<div class="detail-point"><strong>置信度：</strong>${normalizeConfidence(conf)}</div>`;
            if (resultData.decision_logic) {
                html += `<div class="detail-point">${this.escapeHtml(resultData.decision_logic)}</div>`;
            }
            html += `</div>`;

            const strategyNames = {
                'trend_following': '趋势跟踪',
                'mean_reversion': '均值回归',
                'momentum': '动量策略',
                'volatility': '波动率策略',
                'statistical_arbitrage': '统计套利'
            };

            html += `<div class="detail-section"><div class="detail-section-title">各策略信号</div>`;
            for (const [key, sigData] of Object.entries(signals)) {
                const name = strategyNames[key] || key;
                const sig = sigData.signal || '-';
                const confStr = sigData.confidence != null ? sigData.confidence : '-';
                html += `<div class="detail-point"><strong>${name}：</strong>${this.getSignalText(sig)}，置信度${confStr}</div>`;
            }
            html += `</div>`;

            if (resultData.macd_value != null || resultData.rsi_value != null) {
                html += `<div class="detail-section"><div class="detail-section-title">技术指标</div>`;
                if (resultData.macd_value != null) html += `<div class="detail-point"><strong>MACD：</strong>${resultData.macd_value.toFixed(4)}</div>`;
                if (resultData.rsi_value != null) html += `<div class="detail-point"><strong>RSI(14)：</strong>${resultData.rsi_value.toFixed(2)}</div>`;
                html += `</div>`;
            }

            if (prices.length > 0) {
                html += this.createKlineChart(prices, 'technical-kline');
            }

            return html;
        }

        if (agentName === 'expectation_diff_agent') {
            const forecast = resultData.earnings_forecast || {};
            const rating = resultData.research_rating || {};
            const diff = resultData.expectation_diff || {};
            let html = '';

            html += `<div class="detail-section"><div class="detail-section-title">决策逻辑</div>`;
            html += `<div class="detail-point"><strong>综合信号：</strong>${this.getSignalText(resultData.signal)}</div>`;
            const conf = typeof resultData.confidence === 'number' ? resultData.confidence : parseFloat(resultData.confidence || 0);
            html += `<div class="detail-point"><strong>置信度：</strong>${normalizeConfidence(conf)}</div>`;
            if (resultData.decision_logic) {
                html += `<div class="detail-point">${this.escapeHtml(resultData.decision_logic)}</div>`;
            }
            html += `</div>`;

            html += `<div class="detail-section"><div class="detail-section-title">业绩预告</div>`;
            html += `<div class="detail-point"><strong>信号：</strong>${this.getSignalText(forecast.signal || '-')}</div>`;
            if (forecast.details) html += `<div class="detail-point">${this.escapeHtml(forecast.details)}</div>`;
            html += `</div>`;

            html += `<div class="detail-section"><div class="detail-section-title">研报评级</div>`;
            html += `<div class="detail-point"><strong>信号：</strong>${this.getSignalText(rating.signal || '-')}</div>`;
            if (rating.details) html += `<div class="detail-point">${this.escapeHtml(rating.details)}</div>`;
            html += `</div>`;

            if (diff.analyst_expectation !== undefined) {
                html += `<div class="detail-section"><div class="detail-section-title">预期差</div>`;
                html += `<div class="detail-point"><strong>分析师预期：</strong>${diff.analyst_expectation}</div>`;
                html += `<div class="detail-point"><strong>实际vs预期：</strong>${diff.actual_vs_expected || '-'}</div>`;
                html += `</div>`;
            }

            return html;
        }

        if (agentName === 'macro_analyst_agent') {
            const envMap = {'favorable': '有利', 'unfavorable': '不利', 'neutral': '中性', '有利': '有利', '不利': '不利', '中性': '中性'};
            const impactMap = {'positive': '正面', 'negative': '负面', 'neutral': '中性', '正面': '正面', '负面': '负面', '中性': '中性'};
            const decision = resultData.decision_result || {};
            let html = '';

            // 汇总信息
            html += `<div class="detail-section"><div class="detail-section-title">分析结论</div>`;
            html += `<div class="detail-point"><strong>宏观环境：</strong>${envMap[resultData.macro_environment] || resultData.macro_environment || '-'}</div>`;
            html += `<div class="detail-point"><strong>对股票影响：</strong>${impactMap[resultData.impact_on_stock] || resultData.impact_on_stock || '-'}</div>`;
            html += `<div class="detail-point"><strong>信号：</strong>${this.getSignalText(resultData.signal)}</div>`;
            const conf = typeof resultData.confidence === 'number' ? resultData.confidence : parseFloat(resultData.confidence || 0);
            html += `<div class="detail-point"><strong>置信度：</strong>${normalizeConfidence(conf)}</div>`;
            html += `</div>`;

            // 决策逻辑（详细）
            if (resultData.decision_logic) {
                html += `<div class="detail-section"><div class="detail-section-title">决策逻辑</div>`;
                html += `<div class="detail-point" style="white-space: pre-wrap;">${this.escapeHtml(resultData.decision_logic)}</div>`;
                html += `</div>`;
            }

            // 决策结果
            if (decision && Object.keys(decision).length > 0) {
                html += `<div class="detail-section"><div class="detail-section-title">决策结果</div>`;
                if (decision.recommendation) {
                    html += `<div class="detail-point"><strong>建议：</strong>${this.escapeHtml(decision.recommendation)}</div>`;
                }
                if (decision.action) {
                    html += `<div class="detail-point"><strong>操作建议：</strong>${this.escapeHtml(decision.action)}</div>`;
                }
                if (decision.risk_warning) {
                    html += `<div class="detail-point"><strong>风险提示：</strong>${this.escapeHtml(decision.risk_warning)}</div>`;
                }
                html += `</div>`;
            }

            // 关键因素
            const keyFactors = resultData.key_factors || resultData.factors || [];
            if (Array.isArray(keyFactors) && keyFactors.length > 0) {
                html += `<div class="detail-section"><div class="detail-section-title">关键因素 (${keyFactors.length}条)</div>`;
                const items = keyFactors.map((f, i) => {
                    if (typeof f === 'string') {
                        return `<div class="detail-point"><span class="point-num">${i + 1}.</span>${this.escapeHtml(f)}</div>`;
                    } else if (typeof f === 'object' && f.factor) {
                        return `<div class="detail-point"><span class="point-num">${i + 1}.</span>${this.escapeHtml(f.factor)}</div>`;
                    }
                    return '';
                }).join('');
                html += items + `</div>`;
            }

            // 分析详情
            if (resultData.reasoning) {
                html += `<div class="detail-section"><div class="detail-section-title">分析详情</div>`;
                html += `<div class="detail-point" style="white-space: pre-wrap;">${this.escapeHtml(resultData.reasoning.slice(0, 600))}${resultData.reasoning.length > 600 ? '...' : ''}</div>`;
                html += `</div>`;
            }

            return html;
        }

        if (agentName === 'macro_news_agent') {
            const newsResult = resultData.macro_news_analysis_result || resultData;
            let html = '';

            html += `<div class="detail-section"><div class="detail-section-title">📊 新闻概览</div>`;
            html += `<div class="detail-point"><strong>获取数量：</strong>${newsResult.retrieved_news_count || newsResult.获取新闻数 || 0}条</div>`;
            if (newsResult.summary_content || newsResult.新闻摘要) {
                html += `<div class="detail-point"><strong>分析摘要：</strong></div>`;
                html += `<div class="detail-point" style="margin-left: 10px; border-left: 2px solid var(--accent-info); padding-left: 10px;">${this.escapeHtml(newsResult.summary_content || newsResult.新闻摘要)}</div>`;
            }
            html += `</div>`;

            const newsList = resultData.news_list || d.news_list || newsResult.news_list || newsResult.新闻列表 || [];
            if (newsList.length > 0) {
                html += `<div class="detail-section"><div class="detail-section-title">📰 新闻列表</div>`;
                newsList.slice(0, 10).forEach((item, idx) => {
                    const title = item.title || item.标题 || '无标题';
                    const date = item.date || item.发布日期 || '';
                    const source = item.source || item.来源 || '';
                    const sentiment = item.sentiment || item.情绪 || '';
                    const sentimentColor = sentiment === 'positive' ? 'var(--accent-success)' : sentiment === 'negative' ? 'var(--accent-danger)' : 'var(--text-muted)';
                    html += `<div class="news-item" style="margin-bottom: 12px; padding: 10px; background: var(--bg-input); border-radius: 8px; border-left: 3px solid var(--accent-primary);">`;
                    html += `<div style="font-weight: 500; color: var(--text-primary); margin-bottom: 6px;">${idx + 1}. ${this.escapeHtml(title)}</div>`;
                    html += `<div style="font-size: 12px; color: var(--text-muted);">`;
                    if (date) html += `<span style="margin-right: 12px;">📅 ${date}</span>`;
                    if (source) html += `<span style="margin-right: 12px;"> source:${source}</span>`;
                    if (sentiment) html += `<span style="color: ${sentimentColor};">🏷️ ${sentiment}</span>`;
                    html += `</div></div>`;
                });
                html += `</div>`;
            } else {
                html += `<div class="detail-section"><div class="detail-section-title">📰 新闻列表</div>`;
                html += `<div class="detail-point" style="color: var(--text-muted);">暂无新闻数据</div>`;
                html += `</div>`;
            }

            return html;
        }

        if (agentName === 'portfolio_management_agent') {
            const prices = resultData.prices || [];
            let html = this.createDecisionVisualization(resultData);

            if (prices.length > 0) {
                html += this.createKlineChart(prices, 'portfolio-kline');
            }

            const signals = resultData.signal_summary || resultData.各模块信号汇总 || resultData.agent_signals;
            if (signals) {
                html += `<div class="detail-section"><div class="detail-section-title">各模块信号汇总</div>`;
                if (Array.isArray(signals)) {
                    for (const item of signals) {
                        const name = item.agent_name || item.agent || Object.keys(item)[0] || '-';
                        const sig = item.signal || '';
                        const conf = typeof item.confidence === 'number' ? Math.round(item.confidence * 100) + '%' : (item.confidence || '-');
                        html += `<div class="detail-point"><strong>${this.escapeHtml(name)}：</strong>${this.getSignalText(sig)} （置信度${conf}）</div>`;
                    }
                } else if (typeof signals === 'object') {
                    for (const [key, val] of Object.entries(signals)) {
                        const sig = val.signal || val;
                        html += `<div class="detail-point"><strong>${this.escapeHtml(key)}：</strong>${this.getSignalText(sig)}</div>`;
                    }
                }
                html += `</div>`;
            }

            if (resultData.分析报告) {
                html += `<div class="detail-section"><div class="detail-section-title">详细分析报告</div>`;
                html += `<div class="detail-point" style="white-space: pre-wrap; font-size: 12px;">${this.escapeHtml(resultData.分析报告)}</div>`;
                html += `</div>`;
            }

            return html;
        }

        return '';
    },

    createDetailDataSection(title, data) {
        if (!data || (typeof data === 'object' && Object.keys(data).length === 0)) {
            return '';
        }

        return `
            <div class="agent-detail-block">
                <div class="agent-detail-block-title">${title}</div>
                <pre class="agent-detail-json">${this.escapeHtml(JSON.stringify(data, null, 2))}</pre>
            </div>
        `;
    },

    createSignalCard(signal) {
        const agentName = signal.agent_name || signal.agent || '未知模块';
        const normalizedSignal = signal.signal || 'neutral';
        const confidence = typeof signal.confidence === 'number'
            ? signal.confidence
            : Number(signal.confidence || 0);

        return `
            <div class="signal-card">
                <div class="signal-card-top">
                    <span class="signal-card-name">${this.escapeHtml(agentName)}</span>
                    <span class="agent-signal ${normalizedSignal}">${this.getSignalText(normalizedSignal)}</span>
                </div>
                <div class="signal-card-confidence">置信度 ${Math.round((confidence > 1 ? confidence / 100 : confidence) * 100)}%</div>
            </div>
        `;
    },

    createDetailLogEntry(log) {
        const timestamp = log.timestamp ? this.formatTime(log.timestamp) : '';
        const level = log.level || 'info';
        const message = log.message || '';

        return `
            <div class="agent-detail-log-entry">
                <span class="agent-detail-log-time">${timestamp}</span>
                <span class="agent-detail-log-level ${level}">${level}</span>
                <span class="agent-detail-log-message">${this.escapeHtml(message)}</span>
            </div>
        `;
    },

    getActionText(action) {
        const actionMap = {
            'buy': '买入',
            'sell': '卖出',
            'hold': '持有'
        };
        return actionMap[action] || action;
    },

    createHistoryItem(run) {
        const ticker = run.ticker || '';
        const status = run.status || '';
        const action = run.action || '';
        const confidence = run.confidence || 0;
        const createdAt = run.created_at ? this.formatDate(run.created_at) : '';

        return `
            <div class="history-item-wrap">
                <input type="checkbox" class="history-check" data-run="${run.run_id}" onchange="App.onHistorySelect('${run.run_id}')">
                <div class="history-item" onclick="App.showHistoryDetail('${run.run_id}')">
                    <div class="history-ticker">${ticker}</div>
                    <div class="history-meta">
                        <span>${createdAt}</span>
                        <span>置信度: ${normalizeConfidence(confidence)}</span>
                    </div>
                    ${action ? `<span class="history-action ${action}">${this.getActionText(action)}</span>` : ''}
                </div>
                <button class="history-delete" onclick="App.deleteHistoryRun('${run.run_id}', this)" title="删除">×</button>
            </div>
        `;
    },

    createHistoryHeader() {
        return `
            <div class="history-header-bar">
                <label class="history-select-all">
                    <input type="checkbox" onchange="App.toggleAllHistory(this.checked)">全选
                </label>
                <button class="btn-clear" onclick="App.clearAllHistory()">清空历史</button>
            </div>
        `;
    },

    createHistoryDetail(runId, data) {
        if (!data) return '<div class="log-empty">暂无详情</div>';

        const ticker = data.ticker || runId;
        const status = data.status || '';
        const result = data.result || {};

        return `
            <div class="result-summary">
                <div class="result-item">
                    <div class="result-label">股票</div>
                    <div class="result-value">${ticker}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">状态</div>
                    <div class="result-value">${status}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">置信度</div>
                    <div class="result-value">${Math.round((result.confidence || 0) * 100)}%</div>
                </div>
            </div>
            ${result.action ? `
                <div class="result-reasoning">
                    <h4>最终决策</h4>
                    <p>行动: ${this.getActionText(result.action)} | 数量: ${result.quantity || 0}股 | 置信度: ${Math.round((result.confidence || 0) * 100)}%</p>
                </div>
            ` : ''}
        `;
    },

    createHistoryReplay(runId, flow, agentStates, agents, resultData) {
        const startTime = flow.start_time ? this.formatDate(flow.start_time) : '';
        const endTime = flow.end_time ? this.formatDate(flow.end_time) : '';
        
        // 获取最终决策
        const action = resultData.action || 'unknown';
        const quantity = resultData.quantity || 0;
        const confidence = resultData.confidence || 0;
        const reasoning = resultData.reasoning || '';
        
        // 构建最终决策显示
        const actionText = { buy: '买入', sell: '卖出', hold: '持有' }[action] || action;
        const signalClass = action === 'buy' ? 'success' : action === 'sell' ? 'danger' : 'warning';
        
        const finalDecisionHtml = `
            <div class="final-decision-box">
                <div class="decision-title">最终决策</div>
                <div class="decision-action ${signalClass}">${actionText}</div>
                <div class="decision-quantity">数量: ${quantity}股</div>
                <div class="decision-confidence">置信度: ${normalizeConfidence(confidence)}</div>
                ${reasoning ? `<div class="decision-reasoning">${reasoning}</div>` : ''}
            </div>
        `;
        
        // 构建 Agent 卡片（各模块信号）
        const agentCards = agents.map(sig => {
            const signalText = sig.signal ? this.getSignalText(sig.signal) : '-';
            const conf = normalizeConfidence(sig.confidence);
            return `
                <div class="agent-card status-completed" data-agent="${sig.agent_name}">
                    <div class="agent-header">
                        <span class="agent-name">${AGENT_DISPLAY_NAMES[sig.agent_name] || sig.agent_name}</span>
                        <span class="agent-status-badge completed">✓</span>
                    </div>
                    <div class="agent-signal ${sig.signal || ''}">信号: ${signalText}</div>
                    <div class="agent-confidence">置信度: ${conf}</div>
                </div>
            `;
        }).join('');

        return `
            <div class="history-replay">
                <div class="replay-header">
                    <div class="replay-title">历史回放: ${runId}</div>
                    <div class="replay-time">${startTime} - ${endTime}</div>
                </div>
                ${finalDecisionHtml}
                <div class="replay-section-title">各模块分析结果</div>
                <div class="replay-agents">${agentCards}</div>
            </div>
        `;
    },

    renderHistoryAgentDetail(runId, agentName, agentDetail) {
        if (!agentDetail) return '<div class="empty">暂无详情</div>';
        
        const input = agentDetail.input_state || {};
        const output = agentDetail.output_state || {};
        const reasoning = agentDetail.reasoning || {};
        
        return `
            <div class="detail-shell">
                <div class="detail-header">
                    <span class="detail-name">${AGENT_DISPLAY_NAMES[agentName] || agentName}</span>
                    <span class="detail-badge completed">✓</span>
                </div>
                <div class="detail-result">
                    执行时间: ${agentDetail.execution_time_seconds?.toFixed(2) || 0}s
                </div>
                ${output.messages && output.messages.length ? `
                    <div class="detail-messages">
                        <div class="detail-section-title">输出消息</div>
                        <pre class="detail-json">${this.escapeHtml(JSON.stringify(output.messages.slice(-2), null, 2))}</pre>
                    </div>
                ` : ''}
            </div>
        `;
    },

    formatTime(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch {
            return timestamp;
        }
    },

    formatDate(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        } catch {
            return timestamp;
        }
    },

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    createKlineChart(prices, containerId) {
        if (!prices || prices.length === 0) {
            return '<div class="kline-chart-container"><div class="kline-chart-title">K线图</div><div class="empty">暂无数据</div></div>';
        }

        const container = document.querySelector(`#${containerId}`);
        if (!container) return '';

        const data = prices.slice(-60);
        if (data.length < 2) {
            return '<div class="kline-chart-container"><div class="kline-chart-title">K线图</div><div class="empty">数据不足</div></div>';
        }

        const width = container.clientWidth || 600;
        const height = 200;
        const padding = { top: 20, right: 50, bottom: 30, left: 10 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        const pricesData = data.map(d => ({
            date: d.date,
            open: parseFloat(d.open),
            high: parseFloat(d.high),
            low: parseFloat(d.low),
            close: parseFloat(d.close),
            volume: parseFloat(d.volume || 0)
        }));

        const pricesFlat = pricesData.flatMap(d => [d.high, d.low]);
        const minPrice = Math.min(...pricesFlat);
        const maxPrice = Math.max(...pricesFlat);
        const priceRange = maxPrice - minPrice || 1;

        const maxVolume = Math.max(...pricesData.map(d => d.volume));

        const candleWidth = Math.max(2, (chartWidth / data.length) * 0.7);
        const barWidth = chartWidth / data.length;

        let svg = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">`;

        for (let i = 0; i < data.length; i += 10) {
            const x = padding.left + (i / data.length) * chartWidth;
            svg += `<line class="kline-grid-line" x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}" />`;
        }

        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (i / 4) * chartHeight;
            svg += `<line class="kline-grid-line" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" />`;
            const price = maxPrice - (i / 4) * priceRange;
            svg += `<text class="kline-axis-label" x="${width - padding.right + 5}" y="${y + 4}">${price.toFixed(2)}</text>`;
        }

        svg += `<g class="kline-candles">`;
        pricesData.forEach((d, i) => {
            const x = padding.left + (i + 0.5) * barWidth;
            const isUp = d.close >= d.open;
            const colorClass = isUp ? 'kline-candlestick-up' : 'kline-candlestick-down';

            const highY = padding.top + ((maxPrice - d.high) / priceRange) * chartHeight;
            const lowY = padding.top + ((maxPrice - d.low) / priceRange) * chartHeight;
            const openY = padding.top + ((maxPrice - d.open) / priceRange) * chartHeight;
            const closeY = padding.top + ((maxPrice - d.close) / priceRange) * chartHeight;

            const bodyTop = Math.min(openY, closeY);
            const bodyHeight = Math.max(1, Math.abs(closeY - openY));

            svg += `<line class="${colorClass}" stroke-width="1" x1="${x}" y1="${highY}" x2="${x}" y2="${lowY}" />`;
            svg += `<rect class="${colorClass}" x="${x - candleWidth/2}" y="${bodyTop}" width="${candleWidth}" height="${bodyHeight}" rx="1" />`;
        });
        svg += `</g>`;

        const volumeHeight = chartHeight * 0.2;
        const volumeTop = height - padding.bottom - volumeHeight;
        svg += `<g class="kline-volumes" opacity="0.5">`;
        pricesData.forEach((d, i) => {
            const x = padding.left + (i + 0.5) * barWidth;
            const volHeight = (d.volume / maxVolume) * volumeHeight;
            const isUp = d.close >= d.open;
            svg += `<rect class="${isUp ? 'kline-candlestick-up' : 'kline-candlestick-down'}" x="${x - candleWidth/2}" y="${volumeTop + volumeHeight - volHeight}" width="${candleWidth}" height="${volHeight}" rx="1" />`;
        });
        svg += `</g>`;

        svg += `</svg>`;

        return `
            <div class="kline-chart-container">
                <div class="kline-chart-title">K线走势 (近${data.length}日)</div>
                <div class="kline-chart" id="${containerId}-chart">
                    ${svg}
                </div>
            </div>
        `;
    },

    createDecisionVisualization(resultData) {
        if (!resultData) return '';

        const action = resultData.action || 'hold';
        const quantity = resultData.quantity || 0;
        const confidence = resultData.confidence || 0;
        const reasoning = resultData.reasoning || '';

        const actionMap = { buy: '买入', sell: '卖出', hold: '持有' };
        const actionText = actionMap[action] || '持有';

        // 兼容对象和数组两种格式
        const signalSummary = resultData.signal_summary || resultData.各模块信号汇总 || {};
        const agentSignalsArr = resultData.agent_signals || [];
        let bullishCount = 0, bearishCount = 0, neutralCount = 0;

        if (Array.isArray(agentSignalsArr) && agentSignalsArr.length > 0) {
            agentSignalsArr.forEach(s => {
                const sig = (s.signal || '').toLowerCase();
                if (sig === 'bullish' || sig === 'buy' || sig === 'positive') bullishCount++;
                else if (sig === 'bearish' || sig === 'sell' || sig === 'negative' || sig === 'reduce') bearishCount++;
                else neutralCount++;
            });
        } else {
            bullishCount = Object.values(signalSummary).filter(s => s.signal === 'bullish' || s === 'bullish').length;
            bearishCount = Object.values(signalSummary).filter(s => s.signal === 'bearish' || s === 'bearish').length;
            neutralCount = Object.values(signalSummary).filter(s => s.signal === 'neutral' || s === 'neutral').length;
        }

        const normalizedConf = confidence > 1 ? confidence / 100 : confidence;
        const signalLevel = normalizedConf > 0.7 ? 'high' : normalizedConf > 0.4 ? 'medium' : 'low';
        const signalPercent = Math.round(normalizedConf * 100);

        const actionIcons = { buy: '📈', sell: '📉', hold: '⏸️' };

        return `
            <div class="decision-visualization">
                <div class="decision-visualization-title">
                    <span class="icon">${actionIcons[action]}</span>
                    <span>投资决策</span>
                </div>

                <div class="decision-action-box ${action}">
                    <div class="decision-action-main">
                        <div class="decision-action-label">建议操作</div>
                        <div class="decision-action-value ${action}">${actionText}</div>
                    </div>
                    <div class="decision-quantity-box">
                        <div class="decision-quantity-label">建议数量</div>
                        <div class="decision-quantity-value">${quantity}股</div>
                    </div>
                </div>

                <div class="decision-metrics">
                    <div class="decision-metric">
                        <div class="decision-metric-label">置信度</div>
                        <div class="decision-metric-value">${signalPercent}%</div>
                        <div class="decision-metric-bar">
                            <div class="decision-metric-fill ${signalLevel}" style="width: ${signalPercent}%"></div>
                        </div>
                    </div>
                    <div class="decision-metric">
                        <div class="decision-metric-label">看多信号</div>
                        <div class="decision-metric-value" style="color: var(--accent-success)">${bullishCount}</div>
                        <div class="decision-metric-bar">
                            <div class="decision-metric-fill high" style="width: ${(bullishCount/15*100).toFixed(0)}%"></div>
                        </div>
                    </div>
                    <div class="decision-metric">
                        <div class="decision-metric-label">看空信号</div>
                        <div class="decision-metric-value" style="color: var(--accent-danger)">${bearishCount}</div>
                        <div class="decision-metric-bar">
                            <div class="decision-metric-fill low" style="width: ${(bearishCount/15*100).toFixed(0)}%"></div>
                        </div>
                    </div>
                </div>

                <div class="decision-signal-breakdown">
                    <div class="decision-signal-title">各模块信号分布</div>
                    <div class="decision-signal-grid">
                        ${(() => {
                            const labelMap = {
                                'market_data': '市场数据', 'technical': '技术分析', '技术分析': '技术分析',
                                'fundamentals': '基本面', '基本面分析': '基本面',
                                'sentiment': '情绪', '情绪分析': '情绪',
                                'valuation': '估值', '估值分析': '估值',
                                'industry_cycle': '行业周期', '行业周期': '行业周期',
                                'institutional': '机构持仓', '机构持仓': '机构持仓',
                                'expectation_diff': '预期差', '预期差': '预期差',
                                'macro_news': '宏观新闻', '宏观新闻分析': '宏观新闻',
                                'macro': '宏观', '宏观分析': '宏观分析',
                                'bull_research': '看多研究', '看多研究员': '看多研究',
                                'bear_research': '看空研究', '看空研究员': '看空研究',
                                'debate': '辩论', '辩论室': '辩论',
                                'risk': '风险管理', '风险管理': '风险管理'
                            };
                            const gt = Components.getSignalText.bind(Components);
                            if (Array.isArray(agentSignalsArr) && agentSignalsArr.length > 0) {
                                return agentSignalsArr.map(item => {
                                    const name = item.agent_name || item.agent || '';
                                    const sig = item.signal || 'neutral';
                                    const label = labelMap[name] || name;
                                    return '<div class="decision-signal-item"><div class="name">' + Components.escapeHtml(label) + '</div><div class="signal ' + sig + '">' + gt(sig) + '</div></div>';
                                }).join('');
                            } else {
                                return Object.entries(signalSummary).map(([name, s]) => {
                                    const sig = s.signal || s;
                                    const label = labelMap[name] || name;
                                    return '<div class="decision-signal-item"><div class="name">' + Components.escapeHtml(label) + '</div><div class="signal ' + sig + '">' + gt(sig) + '</div></div>';
                                }).join('');
                            }
                        })()}
                    </div>
                </div>

                ${reasoning ? `
                    <div class="detail-section" style="margin-top: 16px;">
                        <div class="detail-section-title">决策说明</div>
                        <div class="detail-point">${this.escapeHtml(reasoning)}</div>
                    </div>
                ` : ''}
            </div>
        `;
    }
};

Components.AGENT_DISPLAY_NAMES = AGENT_DISPLAY_NAMES;
Components.AGENT_ALIASES = AGENT_ALIASES;
Components.normalizeAgentName = normalizeAgentName;
window.Components = Components;
