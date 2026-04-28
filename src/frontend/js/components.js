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
        const message = state.message || '该 Agent 暂无最新说明';
        const recentLogs = logs.slice(-30);
        const details = state.details || {};
        const signalText = signal ? this.getSignalText(signal) : (status === 'completed' ? '该节点无明确交易信号' : '待输出');
        const confidenceText = confidence > 0 ? `${Math.round(confidence * 100)}%` : (status === 'completed' ? '未提供' : '等待执行');

        return `
            <div class="agent-detail-shell">
                <div class="agent-detail-top">
                    <div>
                        <div class="agent-detail-name">${displayName}</div>
                        <div class="agent-detail-subtitle">实时查看该 Agent 的执行状态、日志与结果。</div>
                    </div>
                    <span class="agent-status-badge ${status}">${this.getStatusText(status)}</span>
                </div>

                <div class="agent-detail-metrics">
                    <div class="agent-detail-metric">
                        <div class="agent-detail-label">当前状态</div>
                        <div class="agent-detail-value">${this.getStatusText(status)}</div>
                    </div>
                    <div class="agent-detail-metric">
                        <div class="agent-detail-label">信号判断</div>
                        <div class="agent-detail-value ${signal || ''}">${signalText}</div>
                    </div>
                    <div class="agent-detail-metric">
                        <div class="agent-detail-label">置信度</div>
                        <div class="agent-detail-value">${confidenceText}</div>
                    </div>
                </div>

                <div class="agent-detail-block">
                    <div class="agent-detail-block-title">最后消息</div>
                    <div class="agent-detail-message">${this.escapeHtml(message)}</div>
                </div>

                ${details.summary ? `
                    <div class="agent-detail-block">
                        <div class="agent-detail-block-title">结果摘要</div>
                        <div class="agent-detail-message">${this.escapeHtml(details.summary)}</div>
                    </div>
                ` : ''}

                ${this.createDetailDataSection('输入摘要', details.input)}
                ${this.createDetailDataSection('输出摘要', details.output)}
                ${this.createDetailDataSection('推理结果', details.reasoning)}
                ${this.createDetailDataSection('结构化结果', details.result)}

                <div class="agent-detail-block">
                    <div class="agent-detail-block-title">最近日志</div>
                    <div class="agent-detail-log-list">
                        ${recentLogs.length ? recentLogs.map((log) => this.createDetailLogEntry(log)).join('') : '<div class="agent-detail-empty-inline">该 Agent 还没有产生日志。</div>'}
                    </div>
                </div>
            </div>
        `;
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
            <div class="history-item" onclick="App.showHistoryDetail('${run.run_id}')">
                <div class="history-ticker">${ticker}</div>
                <div class="history-meta">
                    <span>${createdAt}</span>
                    <span>置信度: ${Math.round(confidence * 100)}%</span>
                </div>
                ${action ? `<span class="history-action ${action}">${this.getActionText(action)}</span>` : ''}
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
