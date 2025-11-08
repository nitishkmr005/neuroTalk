/**
 * Real-time audio capture and WebSocket communication for InstaVoice.
 * 
 * This module handles:
 * - Real-time microphone access
 * - Audio chunk processing and streaming
 * - WebSocket communication with the server
 * - Real-time transcription display
 */

class RealTimeAudioProcessor {
    constructor(websocketUrl = 'ws://localhost:8765') {
        this.websocketUrl = websocketUrl;
        this.websocket = null;
        this.mediaStream = null;
        this.audioContext = null;
        this.processor = null;
        this.isRecording = false;
        this.isConnected = false;
        
        // Audio settings
        this.sampleRate = 16000;
        this.chunkSize = 1024;
        this.bufferSize = 4096;
        
        // Callbacks
        this.onTranscription = null;
        this.onAudioProcessed = null;
        this.onStats = null;
        this.onError = null;
        this.onConnectionChange = null;
        
        // Statistics
        this.stats = {
            totalChunks: 0,
            speechChunks: 0,
            maxAmplitude: 0,
            rmsAmplitude: 0,
            bufferDuration: 0
        };
        
        console.log('RealTimeAudioProcessor initialized');
    }
    
    /**
     * Connect to WebSocket server
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            try {
                console.log(`Connecting to WebSocket: ${this.websocketUrl}`);
                
                this.websocket = new WebSocket(this.websocketUrl);
                
                // Set a timeout for connection
                const connectionTimeout = setTimeout(() => {
                    if (!this.isConnected) {
                        reject(new Error('WebSocket connection timeout'));
                    }
                }, 10000); // 10 second timeout
                
                this.websocket.onopen = () => {
                    console.log('WebSocket connected');
                    clearTimeout(connectionTimeout);
                    this.isConnected = true;
                    this.onConnectionChange?.(true);
                    
                    // Initialize processor on server
                    this.sendMessage({
                        type: 'init',
                        settings: {
                            vad_aggressiveness: 1,
                            sample_rate: this.sampleRate,
                            whisper_model: 'base'
                        }
                    });
                    
                    resolve(true);
                };
                
                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.handleServerMessage(data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };
                
                this.websocket.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.isConnected = false;
                    this.onConnectionChange?.(false);
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    clearTimeout(connectionTimeout);
                    const errorMsg = error.message || 'WebSocket connection failed';
                    this.onError?.(errorMsg);
                    reject(new Error(errorMsg));
                };
                
            } catch (error) {
                console.error('Failed to connect WebSocket:', error);
                this.onError?.(`Connection failed: ${error.message}`);
                reject(error);
            }
        });
    }
    
    /**
     * Handle messages from server
     */
    handleServerMessage(data) {
        switch (data.type) {
            case 'init_response':
                console.log('Server initialized:', data);
                break;
                
            case 'audio_processed':
                this.updateStats(data);
                this.onAudioProcessed?.(data);
                break;
                
            case 'transcription':
                console.log('Transcription received:', data.text);
                this.onTranscription?.(data);
                break;
                
            case 'stats':
                this.stats = { ...this.stats, ...data };
                this.onStats?.(this.stats);
                break;
                
            case 'error':
                console.error('Server error:', data.error);
                this.onError?.(data.error);
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    /**
     * Update local statistics
     */
    updateStats(data) {
        this.stats.totalChunks = data.total_chunks || this.stats.totalChunks;
        this.stats.speechChunks = data.speech_chunks || this.stats.speechChunks;
        this.stats.maxAmplitude = Math.max(this.stats.maxAmplitude, data.max_amplitude || 0);
        this.stats.rmsAmplitude = data.rms_amplitude || this.stats.rmsAmplitude;
        this.stats.bufferDuration = data.buffer_duration || this.stats.bufferDuration;
    }
    
    /**
     * Send message to server
     */
    sendMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
            this.onError?.('WebSocket not connected');
        }
    }
    
    /**
     * Start real-time audio capture
     */
    async startRecording() {
        try {
            console.log('Starting real-time audio capture...');
            
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.sampleRate,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });
            
            // Create audio source
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create script processor for real-time processing
            this.processor = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
            
            this.processor.onaudioprocess = (event) => {
                if (this.isRecording) {
                    this.processAudioBuffer(event.inputBuffer);
                }
            };
            
            // Connect audio nodes
            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            console.log('Real-time audio capture started');
            
            return true;
            
        } catch (error) {
            console.error('Failed to start audio capture:', error);
            this.onError?.(`Microphone access failed: ${error.message}`);
            return false;
        }
    }
    
    /**
     * Process audio buffer and send to server
     */
    processAudioBuffer(buffer) {
        try {
            // Get audio data from buffer
            const audioData = buffer.getChannelData(0);
            
            // Convert float32 to int16
            const int16Array = new Int16Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                int16Array[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
            }
            
            // Convert to base64
            const audioBytes = new Uint8Array(int16Array.buffer);
            const base64Audio = btoa(String.fromCharCode.apply(null, audioBytes));
            
            // Send to server
            this.sendMessage({
                type: 'audio_chunk',
                audio_data: base64Audio
            });
            
        } catch (error) {
            console.error('Error processing audio buffer:', error);
        }
    }
    
    /**
     * Stop audio capture
     */
    stopRecording() {
        console.log('Stopping audio capture...');
        
        this.isRecording = false;
        
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        console.log('Audio capture stopped');
    }
    
    /**
     * Reset processor state
     */
    reset() {
        this.sendMessage({ type: 'reset' });
        
        this.stats = {
            totalChunks: 0,
            speechChunks: 0,
            maxAmplitude: 0,
            rmsAmplitude: 0,
            bufferDuration: 0
        };
    }
    
    /**
     * Get current statistics
     */
    getStats() {
        this.sendMessage({ type: 'get_stats' });
        return this.stats;
    }
    
    /**
     * Disconnect and cleanup
     */
    disconnect() {
        this.stopRecording();
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.isConnected = false;
        console.log('RealTimeAudioProcessor disconnected');
    }
    
    /**
     * Check if browser supports required features
     */
    static isSupported() {
        return !!(
            navigator.mediaDevices &&
            navigator.mediaDevices.getUserMedia &&
            window.AudioContext &&
            window.WebSocket
        );
    }
}

/**
 * Streamlit component integration
 */
class StreamlitAudioComponent {
    constructor() {
        this.processor = null;
        this.isInitialized = false;
        
        // Bind to Streamlit if available
        if (window.Streamlit) {
            window.Streamlit.setComponentReady();
        }
    }
    
    /**
     * Initialize the component
     */
    async initialize(config = {}) {
        try {
            if (!RealTimeAudioProcessor.isSupported()) {
                throw new Error('Browser does not support required features');
            }
            
            const websocketUrl = config.websocketUrl || 'ws://localhost:8765';
            this.processor = new RealTimeAudioProcessor(websocketUrl);
            
            // Set up callbacks
            this.processor.onTranscription = (data) => {
                this.sendToStreamlit('transcription', data);
            };
            
            this.processor.onAudioProcessed = (data) => {
                this.sendToStreamlit('audio_processed', data);
            };
            
            this.processor.onStats = (stats) => {
                this.sendToStreamlit('stats', stats);
            };
            
            this.processor.onError = (error) => {
                this.sendToStreamlit('error', { message: error });
            };
            
            this.processor.onConnectionChange = (connected) => {
                this.sendToStreamlit('connection_change', { connected });
            };
            
            // Connect to WebSocket
            await this.processor.connectWebSocket();
            
            this.isInitialized = true;
            console.log('StreamlitAudioComponent initialized');
            
            return true;
            
        } catch (error) {
            console.error('Failed to initialize component:', error);
            this.sendToStreamlit('error', { message: error.message });
            return false;
        }
    }
    
    /**
     * Send data to Streamlit
     */
    sendToStreamlit(eventType, data) {
        if (window.Streamlit) {
            window.Streamlit.setComponentValue({
                event: eventType,
                data: data,
                timestamp: Date.now()
            });
        } else {
            // Fallback for testing
            console.log('Streamlit event:', eventType, data);
        }
    }
    
    /**
     * Start recording
     */
    async startRecording() {
        if (!this.isInitialized || !this.processor) {
            throw new Error('Component not initialized');
        }
        
        return await this.processor.startRecording();
    }
    
    /**
     * Stop recording
     */
    stopRecording() {
        if (this.processor) {
            this.processor.stopRecording();
        }
    }
    
    /**
     * Reset processor
     */
    reset() {
        if (this.processor) {
            this.processor.reset();
        }
    }
    
    /**
     * Get statistics
     */
    getStats() {
        return this.processor ? this.processor.getStats() : {};
    }
    
    /**
     * Cleanup
     */
    cleanup() {
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        this.isInitialized = false;
    }
}

// Global component instance
window.streamlitAudioComponent = new StreamlitAudioComponent();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        RealTimeAudioProcessor,
        StreamlitAudioComponent
    };
}
