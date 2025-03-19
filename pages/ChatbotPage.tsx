import React, { useState, useRef, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { SendHorizontal, ArrowLeft, Trash2, Download, Bot, UserCircle } from 'lucide-react';
import { generateGeminiResponse } from '../services/geminiService';
import type { Message, ChatLocationState } from '../types/chat';


function ChatbotPage() {
    const location = useLocation();
    const navigate = useNavigate();
    const { initialMessage, fromOCR } = (location.state as ChatLocationState) || { 
      initialMessage: '', 
      fromOCR: false 
    };
    
    const [messages, setMessages] = useState<Message[]>([
      { 
        role: 'bot', 
        content: 'Hello! I\'m Drishti AI. How can I help you today?',
        timestamp: new Date().toISOString()
      }
    ]);
  const [inputText, setInputText] = useState(initialMessage || '');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
    
  useEffect(() => {
    if (fromOCR && initialMessage) {
      setMessages(prev => [
        ...prev,
        { 
          role: 'bot', 
          content: 'I received the text from your OCR scan. What would you like to know about it?',
          timestamp: new Date().toISOString()
        }
      ]);
    }
  }, [fromOCR, initialMessage]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;
    
    const userMessage: Message = {
      role: 'user',
      content: inputText,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      const botResponse = await generateGeminiResponse(inputText);
      setMessages(prev => [
        ...prev,
        {
          role: 'bot',
          content: botResponse,
          timestamp: new Date().toISOString()
        }
      ]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev,
        {
          role: 'bot',
          content: 'Sorry, I encountered an error processing your request. Please try again.',
          timestamp: new Date().toISOString()
        }
      ]);
    } finally {
      setIsLoading(false);
    }
    
    setInputText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'bot',
        content: 'Hello! I\'m Drishti AI. How can I help you today?',
        timestamp: new Date().toISOString()
      }
    ]);
    setInputText('');
  };

  const handleDownloadChat = () => {
    const chatContent = messages
      .map(msg => `[${msg.timestamp}] ${msg.role === 'bot' ? 'Drishti AI' : 'You'}: ${msg.content}`)
      .join('\n\n');
    
    const blob = new Blob([chatContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drishti-chat-${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <header className="bg-gray-800/90 backdrop-blur-sm border-b border-gray-700 sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button 
                onClick={() => navigate(-1)} 
                className="p-2 text-gray-400 hover:text-purple-400 transition-colors"
              >
                <ArrowLeft className="h-5 w-5" />
              </button>
              <h1 className="text-xl font-semibold text-white flex items-center">
                <Bot className="h-5 w-5 text-purple-400 mr-2" />
                Drishti AI Assistant
              </h1>
            </div>
            <div className="flex space-x-2">
              <button 
                onClick={clearChat}
                className="p-2 text-gray-400 hover:text-purple-400 transition-colors"
                title="Clear chat"
              >
                <Trash2 className="h-5 w-5" />
              </button>
              <button 
                onClick={handleDownloadChat}
                className="p-2 text-gray-400 hover:text-purple-400 transition-colors"
                title="Download chat"
              >
                <Download className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-lg p-4 mb-4 h-[calc(100vh-220px)] overflow-y-auto">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div 
                  key={index} 
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div 
                    className={`max-w-[80%] p-3 rounded-lg ${
                      message.role === 'user' 
                        ? 'bg-purple-600/80 text-white' 
                        : 'bg-gray-700/80 text-gray-200'
                    }`}
                  >
                    <div className="flex items-center mb-1">
                      {message.role === 'bot' ? (
                        <Bot className="h-4 w-4 text-purple-400 mr-2" />
                      ) : (
                        <UserCircle className="h-4 w-4 text-purple-300 mr-2" />
                      )}
                      <span className="text-xs font-medium">
                        {message.role === 'bot' ? 'Drishti AI' : 'You'}
                      </span>
                    </div>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-700/80 text-gray-200 p-3 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <Bot className="h-4 w-4 text-purple-400" />
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-lg p-3">
            <div className="flex items-end">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your message here..."
                className="flex-grow bg-gray-700/50 text-white rounded-lg p-3 resize-none focus:outline-none focus:ring-2 focus:ring-purple-500 placeholder-gray-400 min-h-[70px] max-h-[200px]"
                rows={1}
              />
              <button
                onClick={handleSendMessage}
                disabled={isLoading || !inputText.trim()}
                className="ml-2 p-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
              >
                <SendHorizontal className="h-5 w-5" />
              </button>
            </div>
            <div className="mt-2 text-xs text-gray-400 text-center">
              Press Enter to send. Shift+Enter for new line.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatbotPage;