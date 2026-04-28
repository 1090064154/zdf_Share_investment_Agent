const API_BASE = '/api';

class SSEConnection {
    constructor(runId, onMessage, onError, onOpen) {
        this.runId = runId;
        this.onMessage = onMessage;
        this.onError = onError;
        this.onOpen = onOpen;
        this.eventSource = null;
        this.connected = false;
        this.closedManually = false;
        this.reconnectTimer = null;
    }

    connect() {
        this.closedManually = false;
        const url = `${API_BASE}/run/${this.runId}/stream`;
        this.eventSource = new EventSource(url);

        this.eventSource.onopen = () => {
            console.log('SSE连接已建立');
            this.connected = true;
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }
            if (this.onOpen) {
                this.onOpen();
            }
        };

        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type !== 'heartbeat') {
                    this.onMessage(data);
                }
            } catch (e) {
                console.error('解析SSE消息失败:', e);
            }
        };

        this.eventSource.onerror = (error) => {
            console.error('SSE错误:', error);
            this.connected = false;
            this.onError(error);
            if (!this.closedManually && !this.reconnectTimer) {
                this.reconnectTimer = setTimeout(() => {
                    this.reconnectTimer = null;
                    this.reconnect();
                }, 3000);
            }
        };
    }

    reconnect() {
        if (!this.connected && !this.closedManually) {
            console.log('尝试重新连接...');
            this.close();
            this.connect();
        }
    }

    close() {
        this.closedManually = true;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
            this.connected = false;
        }
    }
}

const Api = {
    async createRun(params) {
        const response = await fetch(`${API_BASE}/run`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '创建任务失败');
        }
        return response.json();
    },

    async getRun(runId) {
        const response = await fetch(`${API_BASE}/run/${runId}`);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '获取任务状态失败');
        }
        return response.json();
    },

    async startRun(runId) {
        const response = await fetch(`${API_BASE}/run/${runId}/start`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '启动任务失败');
        }
        return response.json();
    },

    async getRuns(page = 1, limit = 20) {
        const response = await fetch(`${API_BASE}/runs?page=${page}&limit=${limit}`);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '获取历史列表失败');
        }
        return response.json();
    },

    async getResult(runId) {
        const response = await fetch(`${API_BASE}/run/${runId}/result`);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '获取任务结果失败');
        }
        return response.json();
    },

    async getAgentLogs(runId, agentName) {
        const response = await fetch(`${API_BASE}/run/${runId}/logs/${agentName}`);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '获取Agent日志失败');
        }
        return response.json();
    },

    createSSEConnection(runId, onMessage, onError, onOpen) {
        const conn = new SSEConnection(runId, onMessage, onError, onOpen);
        conn.connect();
        return conn;
    }
};

window.Api = Api;
window.SSEConnection = SSEConnection;
