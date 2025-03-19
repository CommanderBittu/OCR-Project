export interface Message {
    role: 'user' | 'bot';
    content: string;
    timestamp: string;
  }
  
  export interface ChatLocationState {
    initialMessage?: string;
    fromOCR?: boolean;
  }