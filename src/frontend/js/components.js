const AGENT_DESCRIPTIONS = {
    'market_data_agent': '收集股票的历史价格数据、成交量、行业分类和市场整体状况，为后续分析提供基础数据支撑。',
    'technical_analyst_agent': '基于价格和成交量数据，运用趋势跟踪、均值回归、动量策略等技术指标判断短期走势。',
    'fundamentals_agent': '分析公司的盈利能力、成长性和财务健康状况，评估基本面质量。',
    'sentiment_agent': '通过新闻、研报和社交媒体等渠道，分析市场参与者的情绪倾向。',
    'valuation_agent': '评估股票当前价格与合理价值之间的关系，计算折扣率判断估值高低。',
    'industry_cycle_agent': '分析所处行业的周期阶段和景气度，判断行业整体趋势。',
    'institutional_agent': '追踪机构投资者的持仓变化和买卖动向，捕捉大资金的方向。',
    'expectation_diff_agent': '比较市场预期与实际业绩表现，挖掘预期差带来的投资机会。',
    'macro_news_agent': '收集影响市场的宏观新闻和政策动向，分析对股票的直接或间接影响。',
    'macro_analyst_agent': '综合宏观环境分析，研判宏观经济对个股的影响程度。',
    'researcher_bull_agent': '汇总九维分析中偏正向的信号，形成系统性的看多逻辑和投资依据。',
    'researcher_bear_agent': '汇总九维分析中偏负向的信号，形成系统性的看空逻辑和风险提示。',
    'debate_room_agent': '组织多空双方观点进行辩论，形成最终的市场判断结论。',
    'risk_management_agent': '评估整体风险水平，确定最大持仓规模和风险控制建议。',
    'portfolio_management_agent': '综合所有分析结果，给出最终的交易动作、数量和置信度。'
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
                        置信度: ${Math.round(confidence * 100)}%
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
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
                    <div class="result-value">${Math.round(confidence * 100)}%</div>
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
        lines.push(`最终建议为${actionText}，建议数量 ${quantity} 股，系统整体把握度约 ${Math.round(confidence * 100)}%。`);

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
        const confidenceText = confidence > 0 ? `${Math.round(confidence * 100)}%` : '-';
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
                ${summary ? `<div class="detail-summary">${this.escapeHtml(summary)}</div>` : ''}
                ${resultText ? `<div class="detail-result">${resultText}</div>` : ''}
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
            const trend = signals.trend_following?.signal || '-';
            const mean = signals.mean_reversion?.signal || '-';
            const mom = signals.momentum?.signal || '-';
            return `技术面：趋势跟踪[${trend}]，均值回归[${mean}]，动量策略[${mom}]`;
        }
        if (agentName === 'fundamentals_agent') {
            const r = d.reasoning || {};
            const profit = r.profitability_signal?.signal || '-';
            const growth = r.growth_signal?.signal || '-';
            const health = r.financial_health_signal?.signal || '-';
            return `基本面：盈利能力[${profit}]，成长性[${growth}]，财务健康[${health}]`;
        }
        if (agentName === 'sentiment_agent') {
            return d.reasoning ? d.reasoning.slice(0, 100) : '情绪分析暂无详情';
        }
        if (agentName === 'valuation_agent') {
            const price = d.current_price || '-';
            const fair = d.fair_value || '-';
            const discount = d.discount_rate || '-';
            return `估值：当前价${price}元，合理价${fair}元，折扣率${discount}`;
        }
        if (agentName === 'macro_analyst_agent') {
            return `���观环境：${d.macro_environment || '-'}，对股票影响：${d.impact_on_stock || '-'}`;
        }
        if (agentName === 'macro_news_agent') {
            return d.reasoning ? d.reasoning.slice(0, 120) : '暂无宏观新闻分析';
        }
        if (agentName === 'risk_management_agent') {
            return `风险评分：${d.风险评分 || '-'}，建议：${d.交易行动 || 'hold'}，最大持仓：${d.最大持仓规模 || '-'}`;
        }
        if (agentName === 'portfolio_management_agent') {
            return `最终决策：${this.getActionText(d.action || 'hold')} ${d.quantity || 0}股`;
        }
        if (agentName === 'industry_cycle_agent') {
            return `行业周期：${d.行业周期 || '-'}，阶段：${d.周期阶段 || '-'}`;
        }
        if (agentName === 'institutional_agent') {
            const holders = d机构持仓变化 || '-';
            return `机构持仓：${holders}`;
        }
        if (agentName === 'expectation_diff_agent') {
            return `预期差：${d.预期差 || '-'}，方向：${d.方向 || '-'}`;
        }
        if (agentName === 'researcher_bull_agent') {
            return d.综合结论 ? d.综合结论.slice(0, 100) : '看多研究结论';
        }
        if (agentName === 'researcher_bear_agent') {
            return d.综合结论 ? d.综合结论.slice(0, 100) : '看空研究结论';
        }
        if (agentName === 'debate_room_agent') {
            return `辩论结论：${d.辩论结果 || '-'}，胜负：${d.胜方 || '-'}`;
        }
        if (agentName === 'market_data_agent') {
            const prices = d.prices || [];
            return `获取数据：${prices.length}条价格记录，行业：${d.industry || '-'}`;
        }
        
        return JSON.stringify(data).slice(0, 150);
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
                        <span>置信度: ${Math.round(confidence * 100)}%</span>
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

    createHistoryReplay(runId, flow, agentStates, agents) {
        const startTime = flow.start_time ? this.formatDate(flow.start_time) : '';
        const endTime = flow.end_time ? this.formatDate(flow.end_time) : '';
        
        // 构建 Agent 卡片
        const agentCards = Object.keys(agentStates).map(name => {
            const state = agentStates[name];
            const signalText = state.signal ? this.getSignalText(state.signal) : '-';
            const conf = state.confidence ? `${Math.round(state.confidence * 100)}%` : '-';
            return `
                <div class="agent-card status-completed" data-agent="${name}" onclick="App.selectHistoryAgent('${runId}', '${name}')">
                    <div class="agent-header">
                        <span class="agent-name">${AGENT_DISPLAY_NAMES[name] || name}</span>
                        <span class="agent-status-badge completed">✓</span>
                    </div>
                    <div class="agent-signal ${state.signal || ''}">信号: ${signalText}</div>
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
                <div class="replay-agents">${agentCards}</div>
                <div class="replay-detail" id="historyAgentDetail">
                    <div class="empty">点击左侧卡片查看详情</div>
                </div>
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
    }
};

Components.AGENT_DISPLAY_NAMES = AGENT_DISPLAY_NAMES;
Components.AGENT_ALIASES = AGENT_ALIASES;
Components.normalizeAgentName = normalizeAgentName;
window.Components = Components;
