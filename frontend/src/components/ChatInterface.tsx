'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Menu, Plus, MessageSquare, Edit3 } from 'lucide-react'

interface Message {
  id: string
  content: string
  isUser: boolean
  timestamp: Date
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
}

export default function ChatInterface() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [currentConversation?.messages])

  const createNewConversation = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date()
    }
    setConversations(prev => [newConversation, ...prev])
    setCurrentConversation(newConversation)
  }

  const sendMessage = async () => {
    if (!input.trim()) return

    let conversation = currentConversation
    if (!conversation) {
      conversation = {
        id: Date.now().toString(),
        title: input.slice(0, 50) + (input.length > 50 ? '...' : ''),
        messages: [],
        createdAt: new Date()
      }
      setConversations(prev => [conversation!, ...prev])
      setCurrentConversation(conversation)
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      isUser: true,
      timestamp: new Date()
    }

    const updatedConversation = {
      ...conversation,
      messages: [...conversation.messages, userMessage]
    }

    setCurrentConversation(updatedConversation)
    setConversations(prev => prev.map(conv => 
      conv.id === conversation.id ? updatedConversation : conv
    ))

    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      })

      if (!response.ok) {
        const errorData = await response.text()
        console.error('Backend error:', errorData)
        throw new Error(`Failed to send message: ${response.status}`)
      }

      const data = await response.json()
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response || data.answer || 'No response received',
        isUser: false,
        timestamp: new Date()
      }

      const finalConversation = {
        ...updatedConversation,
        messages: [...updatedConversation.messages, botMessage]
      }

      setCurrentConversation(finalConversation)
      setConversations(prev => prev.map(conv => 
        conv.id === conversation.id ? finalConversation : conv
      ))
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, there was an error processing your message. Please try again.',
        isUser: false,
        timestamp: new Date()
      }

      const errorConversation = {
        ...updatedConversation,
        messages: [...updatedConversation.messages, errorMessage]
      }

      setCurrentConversation(errorConversation)
      setConversations(prev => prev.map(conv => 
        conv.id === conversation.id ? errorConversation : conv
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex h-screen bg-[#2d3748]">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 overflow-hidden bg-[#374151] border-r border-[#4a5568]`}>
        <div className="p-3 border-b border-[#4a5568]">
          <button
            onClick={createNewConversation}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm font-medium text-[#f7fafc] bg-transparent hover:bg-[#4a5568] rounded-md transition-colors"
          >
            <Plus size={16} />
            New chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {conversations.map((conversation) => (
            <button
              key={conversation.id}
              onClick={() => setCurrentConversation(conversation)}
              className={`w-full px-3 py-2 text-left hover:bg-[#4a5568] rounded-md mx-2 my-1 transition-colors group ${
                currentConversation?.id === conversation.id ? 'bg-[#4a5568]' : ''
              }`}
            >
              <div className="flex items-center gap-2">
                <MessageSquare size={16} className="text-[#a0aec0] flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-[#f7fafc] truncate">
                    {conversation.title}
                  </div>
                </div>
                <Edit3 size={14} className="text-[#a0aec0] opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#4a5568] bg-[#374151]">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-[#4a5568] rounded-md transition-colors"
          >
            <Menu size={20} className="text-[#f7fafc]" />
          </button>
          <h1 className="text-lg font-semibold text-[#f7fafc]">
            FinanceScope
          </h1>
          <div className="w-10"></div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          {!currentConversation?.messages.length && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="w-16 h-16 bg-[#c7700a] rounded-full mx-auto mb-4 flex items-center justify-center">
                  <span className="text-white font-bold text-xl">F</span>
                </div>
                <h2 className="text-2xl font-semibold text-[#f7fafc] mb-2">How can FinanceScope help you today?</h2>
                <p className="text-[#a0aec0] text-sm">Ask me anything about your data, stocks, or financial analysis</p>
              </div>
            </div>
          )}
          
          {currentConversation?.messages.map((message) => (
            <div
              key={message.id}
              className={`border-b border-[#4a5568] ${message.isUser ? 'bg-[#374151]' : 'bg-[#2d3748]'}`}
            >
              <div className="max-w-3xl mx-auto px-4 py-6">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center">
                    {message.isUser ? (
                      <div className="w-8 h-8 bg-[#6b7280] rounded-full flex items-center justify-center">
                        <span className="text-white font-medium text-sm">U</span>
                      </div>
                    ) : (
                      <div className="w-8 h-8 bg-[#c7700a] rounded-full flex items-center justify-center">
                        <span className="text-white font-bold text-sm">F</span>
                      </div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-[#f7fafc] text-base leading-relaxed whitespace-pre-wrap">
                      {message.content}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="border-b border-[#4a5568] bg-[#2d3748]">
              <div className="max-w-3xl mx-auto px-4 py-6">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center">
                    <div className="w-8 h-8 bg-[#c7700a] rounded-full flex items-center justify-center">
                      <span className="text-white font-bold text-sm">F</span>
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-[#c7700a] rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-[#c7700a] rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 bg-[#c7700a] rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-[#4a5568] bg-[#374151]">
          <div className="max-w-3xl mx-auto px-4 py-4">
            <div className="relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message FinanceScope..."
                className="w-full px-4 py-3 pr-12 border border-[#4a5568] rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-[#c7700a] focus:border-transparent text-[#f7fafc] placeholder-[#a0aec0] bg-[#2d3748]"
                rows={1}
                style={{ minHeight: '50px', maxHeight: '200px' }}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || isLoading}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-[#c7700a] hover:text-[#a0590a] disabled:text-[#9ca3af] transition-colors"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}