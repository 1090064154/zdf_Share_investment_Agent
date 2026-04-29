const App = {
    currentRunId: null,
    sseConnection: null,
    agentStates: {},
    agentLogs: {},
    selectedAgent: null,
    isRunning: false,
    completedAgents: 0,

    init() {
        this.initAgentGrid();
        this.bindEvents();
        this.loadHistory();
    },

    initAgentGrid() {
        const defaultAgents = {
            'market_data_agent': { status: 'pending', details: {} },
            'technical_analyst_agent': { status: 'pending', details: {} },
            'fundamentals_agent': { status: 'pending', details: {} },
            'sentiment_agent': { status: 'pending', details: {} },
            'valuation_agent': { status: 'pending', details: {} },
            'industry_cycle_agent': { status: 'pending', details: {} },
            'institutional_agent': { status: 'pending', details: {} },
            'expectation_diff_agent': { status: 'pending', details: {} },
            'macro_news_agent': { status: 'pending', details: {} },
            'macro_analyst_agent': { status: 'pending', details: {} },
            'researcher_bull_agent': { status: 'pending', details: {} },
            'researcher_bear_agent': { status: 'pending', details: {} },
            'debate_room_agent': { status: 'pending', details: {} },
            'risk_management_agent': { status: 'pending', details: {} },
            'portfolio_management_agent': { status: 'pending', details: {} }
        };

        this.agentStates = defaultAgents;
        this.agentLogs = {};
        Object.keys(defaultAgents).forEach((name) => {
            this.agentLogs[name] = [];
        });
        this.selectedAgent = 'market_data_agent';
        this.completedAgents = 0;
        this.updateProgress();

        const grid = document.getElementById('agentGrid');
        if (grid) {
            grid.innerHTML = Components.createAgentWorkflow(this.getRenderableAgentStates());
        }
        this.renderAgentDetail();
    },

    bindEvents() {
        const tickerInput = document.getElementById('ticker');
        if (tickerInput) {
            tickerInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.executeAnalysis();
                }
            });
        }

        const grid = document.getElementById('agentGrid');
        if (grid) {
            grid.addEventListener('click', (e) => {
                const card = e.target.closest('.agent-card');
                if (!card) return;
                const agentName = card.dataset.agent;
                this.selectAgent(agentName);
            });
        }
    },

    async executeAnalysis() {
        if (this.isRunning) return;

        if (this.sseConnection) {
            this.sseConnection.close();
            this.sseConnection = null;
        }

        const ticker = document.getElementById('ticker')?.value.trim();
        const horizon = document.getElementById('horizon')?.value || 'medium';
        const initialCapital = parseInt(document.getElementById('initialCapital')?.value) || 100000;
        const initialPosition = parseInt(document.getElementById('initialPosition')?.value) || 0;
        const numOfNews = parseInt(document.getElementById('numOfNews')?.value) || 5;
        const startDate = document.getElementById('startDate')?.value || null;
        const endDate = document.getElementById('endDate')?.value || null;

        if (!ticker) {
            this.showToast('请输入股票代码', 'error');
            return;
        }

        this.setInputsDisabled(true);
        this.resetState();
        this.updateLiveIndicator(true, '连接中');
        this.appendLog('system', 'info', '正在创建任务...');

        try {
            const params = {
                ticker,
                investment_horizon: horizon,
                initial_capital: initialCapital,
                initial_position: initialPosition,
                show_reasoning: true,
                num_of_news: numOfNews,
                start_date: startDate || null,
                end_date: endDate || null
            };

            const runData = await Api.createRun(params);
            this.currentRunId = runData.run_id;
            this.appendLog('system', 'info', `任务已创建: ${this.currentRunId}`);

            // 建立SSE连接
            this.sseConnection = Api.createSSEConnection(
                this.currentRunId,
                (data) => this.handleSSEEvent(data),
                (error) => this.handleSSEError(error),
                () => {
                    this.updateLiveIndicator(true, '已连接');
                    this.appendLog('system', 'info', 'SSE连接已建立，等待实时日志...');
                }
            );

            // 启动任务
            const startData = await Api.startRun(this.currentRunId);
            this.isRunning = true;
            this.updateButtons();
            this.appendLog('system', 'info', '任务已启动，正在分析...');

        } catch (error) {
            this.showToast(`执行失败: ${error.message}`, 'error');
            this.setInputsDisabled(false);
            this.updateLiveIndicator(false);
        }
    },

    stopAnalysis() {
        if (this.sseConnection) {
            this.sseConnection.close();
            this.sseConnection = null;
        }
        this.isRunning = false;
        this.currentRunId = null;
        this.setInputsDisabled(false);
        this.updateButtons();
        this.updateLiveIndicator(false);
        this.showToast('任务已停止', 'info');
    },

    handleSSEEvent(data) {
        console.log('SSE收到:', data);
        
        var type = data.type;
        var agentName = Components.normalizeAgentName(data.agent);

        switch (type) {
            case 'system_status':
                this.appendLog('system', 'info', data.message || '系统状态更新');
                break;

            case 'agent_start':
                this.updateAgentState(agentName, { status: 'running', message: data.message });
                this.addAgentLog(agentName, data.level || 'info', data.message || '开始执行');
                this.appendLog(agentName, 'info', `开始: ${data.message || ''}`);
                break;

            case 'agent_log':
                if (agentName && this.agentStates[agentName]) {
                    this.updateAgentState(agentName, { message: data.message || this.agentStates[agentName].message });
                }
                this.addAgentLog(agentName, data.level || 'info', data.message || '');
                this.appendLog(agentName, data.level || 'info', data.message || '');
                break;

            case 'agent_complete':
                this.updateAgentState(agentName, {
                    status: data.status === 'failed' ? 'failed' : 'completed',
                    signal: data.signal,
                    confidence: data.confidence || 0,
                    message: data.message,
                    details: data.details || this.agentStates[agentName]?.details || {}
                });
                this.addAgentLog(agentName, data.status === 'failed' ? 'error' : 'success', data.message || '执行完成');
                this.updateProgress();
                var signalText = data.signal ? Components.getSignalText(data.signal) : '待定';
                this.appendLog(agentName, 'success', `完成 - 信号: ${signalText}, 置信度: ${Math.round((data.confidence || 0) * 100)}%`);
                break;

            case 'task_complete':
                this.handleTaskComplete(data.result);
                break;

            case 'task_error':
                this.handleTaskError(data.error);
                break;

            case 'heartbeat':
                break;
                
            default:
                console.log('未知事件类型:', type, data);
        }
    },

    handleSSEError(error) {
        console.error('SSE连接错误:', error);
        if (this.isRunning) {
            this.updateLiveIndicator(true, '重连中');
            this.showToast('连接断开，尝试重新连接...', 'error');
        }
    },

    handleTaskComplete(result) {
        this.isRunning = false;
        this.setInputsDisabled(false);
        this.updateButtons();
        this.updateLiveIndicator(false, '已完成');

        if (result) {
            const actionText = Components.getActionText(result.action);
            document.getElementById('resultPanel').style.display = 'block';
            document.getElementById('resultSummary').innerHTML = Components.createResultPanel(result);
            this.showToast(`分析完成: ${actionText}`, 'success');
        }
    },

    handleTaskError(error) {
        this.isRunning = false;
        this.setInputsDisabled(false);
        this.updateButtons();
        this.updateLiveIndicator(false, '已失败');
        this.showToast(`任务失败: ${error}`, 'error');
    },

    updateAgentState(agentName, state) {
        if (!agentName) return;
        if (!this.agentStates[agentName]) {
            this.agentStates[agentName] = { status: 'pending' };
        }
        if (!this.agentLogs[agentName]) {
            this.agentLogs[agentName] = [];
        }
        Object.assign(this.agentStates[agentName], state);

        const card = document.querySelector(`.agent-card[data-agent="${agentName}"]`);
        if (card) {
            const newCard = Components.createAgentCard(agentName, this.getRenderableAgentState(agentName));
            card.outerHTML = newCard;
        }

        if (this.selectedAgent === agentName) {
            this.renderAgentDetail();
        }
    },

    addAgentLog(agentName, level, message) {
        if (!agentName || !message) return;
        if (!this.agentLogs[agentName]) {
            this.agentLogs[agentName] = [];
        }
        this.agentLogs[agentName].push({
            timestamp: new Date().toISOString(),
            level: level || 'info',
            message
        });
        if (this.agentLogs[agentName].length > 200) {
            this.agentLogs[agentName] = this.agentLogs[agentName].slice(-200);
        }
        if (this.selectedAgent === agentName) {
            this.renderAgentDetail();
        }
    },

    selectAgent(agentName) {
        if (!agentName) return;
        this.selectedAgent = agentName;

        const grid = document.getElementById('agentGrid');
        if (grid) {
            grid.innerHTML = Components.createAgentWorkflow(this.getRenderableAgentStates());
        }
        this.renderAgentDetail();
    },

    renderAgentDetail() {
        const container = document.getElementById('agentDetailContent');
        if (!container) return;

        const agentName = this.selectedAgent;
        if (!agentName || !this.agentStates[agentName]) {
            container.innerHTML = '<div class="agent-detail-empty">点击上方任一 Agent 卡片，查看它的执行过程与结果。</div>';
            return;
        }

        container.innerHTML = Components.createAgentDetail(
            agentName,
            this.agentStates[agentName],
            this.agentLogs[agentName] || []
        );
    },

    getRenderableAgentState(agentName) {
        return {
            ...this.agentStates[agentName],
            isActive: this.selectedAgent === agentName
        };
    },

    getRenderableAgentStates() {
        const states = {};
        Object.keys(this.agentStates).forEach((agentName) => {
            states[agentName] = this.getRenderableAgentState(agentName);
        });
        return states;
    },

    appendLog(agent, level, message) {
        const container = document.getElementById('logContainer');
        if (!container) return;

        const emptyMsg = container.querySelector('.log-empty');
        if (emptyMsg) {
            emptyMsg.remove();
        }

        const log = { agent, level, message, timestamp: new Date().toISOString() };
        container.insertAdjacentHTML('beforeend', Components.createLogEntry(log));
        container.scrollTop = container.scrollHeight;
    },

    updateProgress() {
        const total = Object.keys(this.agentStates).length;
        const completed = Object.values(this.agentStates).filter((agent) => agent.status === 'completed').length;
        this.completedAgents = completed;
        
        const progressEl = document.getElementById('agentProgress');
        const fillEl = document.getElementById('progressFill');
        
        if (progressEl) {
            progressEl.textContent = `${completed}/${total}`;
        }
        if (fillEl) {
            fillEl.style.width = `${(completed / total) * 100}%`;
        }
    },

    updateLiveIndicator(active, label = null) {
        const indicator = document.getElementById('liveIndicator');
        if (!indicator) return;

        if (active) {
            indicator.innerHTML = `
                <span class="live-dot"></span>
                <span>${label || '连接中'}</span>
            `;
            indicator.style.display = 'flex';
        } else {
            indicator.innerHTML = `
                <span class="live-dot" style="background: var(--text-muted);"></span>
                <span>${label || '已断开'}</span>
            `;
        }
    },

    updateButtons() {
        const executeBtn = document.getElementById('executeBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (executeBtn) executeBtn.disabled = this.isRunning;
        if (stopBtn) stopBtn.disabled = !this.isRunning;
    },

    resetState() {
        this.currentRunId = null;
        this.completedAgents = 0;
        this.initAgentGrid();
        document.getElementById('logContainer').innerHTML = '<div class="log-empty">点击"执行分析"开始...</div>';
        document.getElementById('resultPanel').style.display = 'none';
        document.getElementById('startDate').value = '';
        document.getElementById('endDate').value = '';
        this.updateProgress();
    },

    setInputsDisabled(disabled) {
        const inputs = ['ticker', 'horizon', 'initialCapital', 'initialPosition', 'numOfNews', 'startDate', 'endDate'];
        inputs.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.disabled = disabled;
        });
        this.updateButtons();
    },

    async showHistory() {
        const panel = document.getElementById('historyPanel');
        if (!panel) return;

        if (panel.style.display === 'none') {
            panel.style.display = 'block';
            await this.loadHistory();
        } else {
            panel.style.display = 'none';
        }
    },

    async loadHistory() {
        const list = document.getElementById('historyList');
        if (!list) return;

        try {
            console.log('加载历史记录...');
            const response = await Api.getRuns(1, 20);
            const data = response.items || response;
            console.log('历史记录数据:', data);
            if (data && data.length > 0) {
                list.innerHTML = Components.createHistoryHeader() + data.map(run => Components.createHistoryItem(run)).join('');
            } else {
                list.innerHTML = '<div class="log-empty">暂无历史记录</div>';
            }
        } catch (error) {
            console.error('加载历史失败:', error);
            list.innerHTML = '<div class="log-empty">加载失败</div>';
        }
    },

    async deleteHistoryRun(runId, btn) {
        if (!confirm('确定要删除这条记录吗？')) return;

        btn.disabled = true;

        try {
            console.log('删除记录:', runId);
            const result = await Api.deleteRun(runId);
            console.log('删除结果:', result);
            this.showToast('已删除', 'success');

            setTimeout(async () => {
                await this.loadHistory();
            }, 100);
        } catch (error) {
            console.error('删除失败:', error);
            this.showToast('删除失败: ' + error.message, 'error');
            setTimeout(async () => {
                await this.loadHistory();
            }, 500);
        } finally {
            btn.disabled = false;
        }
    },

    async clearAllHistory() {
        if (!confirm('确定要清空所有历史记录吗？此操作不可恢复！')) return;
        
        try {
            await Api.clearAllRuns();
            this.showToast('已清空', 'success');
            await this.loadHistory();
        } catch (error) {
            this.showToast('清空失败', 'error');
        }
    },

    async showHistoryDetail(runId) {
        const modal = document.getElementById('historyModal');
        const detail = document.getElementById('historyDetail');
        if (!modal || !detail) return;

        try {
            const flow = await Api.getRunFlow(runId);
            const agents = await Api.getRunAgents(runId);
            
            // 构建 Agent 状态用于显示
            const agentStates = {};
            const agentLogs = {};
            
            for (const agent of agents) {
                agentStates[agent.agent_name] = {
                    status: 'completed',
                    signal: null,
                    confidence: 0,
                    message: `${agent.agent_name} 执行完成`
                };
                agentLogs[agent.agent_name] = [];
            }
            
            // 获取每个 Agent 的详情
            for (const agent of agents) {
                try {
                    const agentDetail = await Api.getAgentDetail(runId, agent.agent_name);
                    if (agentDetail.output_state) {
                        const output = agentDetail.output_state;
                        const messages = output.messages || [];
                        if (messages.length > 0) {
                            const lastMsg = messages[messages.length - 1];
                            if (lastMsg && lastMsg.content) {
                                try {
                                    const content = JSON.parse(lastMsg.content);
                                    agentStates[agent.agent_name].signal = content.signal;
                                    agentStates[agent.agent_name].confidence = content.confidence || 0;
                                } catch (e) {}
                            }
                        }
                    }
                } catch (e) {}
            }
            
            // 渲染历史详情
            detail.innerHTML = Components.createHistoryReplay(runId, flow, agentStates, agents);
            modal.classList.add('active');
        } catch (error) {
            detail.innerHTML = '<div class="log-empty">加载详情失败: ' + error.message + '</div>';
            modal.classList.add('active');
        }
    },

    onHistorySelect(runId) {
        // 可以在此处处理多选逻辑
    },

    toggleAllHistory(checked) {
        const checks = document.querySelectorAll('.history-check');
        checks.forEach(cb => cb.checked = checked);
    },

    async selectHistoryAgent(runId, agentName) {
        const container = document.getElementById('historyAgentDetail');
        if (!container) return;
        
        try {
            const detail = await Api.getAgentDetail(runId, agentName);
            container.innerHTML = Components.renderHistoryAgentDetail(agentName, detail);
        } catch (error) {
            container.innerHTML = '<div class="empty">加载失败</div>';
        }
    },

    closeHistoryModal() {
        const modal = document.getElementById('historyModal');
        if (modal) {
            modal.classList.remove('active');
        }
    },

    showSettings() {
        this.showToast('设置功能开发中', 'info');
    },

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
};

document.addEventListener('DOMContentLoaded', () => App.init());
