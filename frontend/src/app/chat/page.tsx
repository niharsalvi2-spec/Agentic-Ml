"use client";

import { useState, useRef, useEffect } from "react";
import gsap from "gsap";
import { Sparkles, ArrowRight, Paperclip, Mic, Code, CheckCircle2, CircleDashed, FileCode2, Copy } from "lucide-react";
import Link from "next/link";
import { useChat } from "./layout";
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

interface AgentMessage {
  agent: string;
  message: string;
}

export default function ChatPage() {
  const { addHistory } = useChat();
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [selectedModel, setSelectedModel] = useState("Gemini 3.5 Flash");
  
  // Artifact State
  const [activeArtifact, setActiveArtifact] = useState<{code: string, language: string} | null>(null);
  
  const titleRef = useRef<HTMLHeadingElement>(null);
  const inputContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const artifactPaneRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    gsap.fromTo(titleRef.current, 
      { opacity: 0, y: 20 }, 
      { opacity: 1, y: 0, duration: 1, ease: "power3.out", delay: 0.1 }
    );
    gsap.fromTo(inputContainerRef.current, 
      { opacity: 0, y: 20 }, 
      { opacity: 1, y: 0, duration: 1, ease: "power3.out", delay: 0.3 }
    );
  }, []);

  // Parse messages for artifacts
  useEffect(() => {
    let foundCode = "";
    let foundLang = "";
    
    // Scan from the newest message backwards to find the active code block
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      const codeMatch = msg.message.match(/```(\w+)?\n([\s\S]*?)```/);
      if (codeMatch) {
        foundLang = codeMatch[1] || "python";
        foundCode = codeMatch[2];
        break; // Only show the most recent code block
      } else {
        // If it's currently streaming a code block (no closing backticks yet)
        const streamingMatch = msg.message.match(/```(\w+)?\n([\s\S]*)/);
        if (streamingMatch && msg.agent !== "User") {
            foundLang = streamingMatch[1] || "python";
            foundCode = streamingMatch[2];
            break;
        }
      }
    }

    if (foundCode && (!activeArtifact || activeArtifact.code !== foundCode)) {
        setActiveArtifact({ code: foundCode, language: foundLang });
        if (!activeArtifact && artifactPaneRef.current) {
            gsap.fromTo(artifactPaneRef.current, 
                { x: 100, opacity: 0 }, 
                { x: 0, opacity: 1, duration: 0.6, ease: "power3.out" }
            );
        }
    } else if (!foundCode && activeArtifact) {
        setActiveArtifact(null);
    }

    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;
    
    const userPrompt = input;
    addHistory(userPrompt);
    setInput("");
    setIsProcessing(true);
    setMessages([{ agent: "User", message: userPrompt }]);
    setActiveArtifact(null); // Clear previous artifacts

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userPrompt, model: selectedModel }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        let parsedMessages: AgentMessage[] = [];
        let isDone = false;
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').replace(/\\n/g, '').trim();
            if (dataStr === '[DONE]') {
              isDone = true;
              break;
            }
            if (dataStr) {
              try {
                const data = JSON.parse(dataStr);
                parsedMessages.push(data);
              } catch (e) {
                console.error("Error parsing JSON chunk:", dataStr);
              }
            }
          }
        }
        
        if (parsedMessages.length > 0) {
          setMessages(prev => {
            const newMessages = [...prev];
            for (const data of parsedMessages) {
              const last = newMessages[newMessages.length - 1];
              if (last && last.agent === data.agent) {
                newMessages[newMessages.length - 1] = { ...last, message: last.message + data.message };
              } else {
                newMessages.push(data);
              }
            }
            return newMessages;
          });
        }
        
        if (isDone) {
          setIsProcessing(false);
          break;
        }
      }
    } catch (error) {
      console.error("Failed to stream:", error);
      setIsProcessing(false);
    }
  };

  const copyToClipboard = () => {
      if(activeArtifact) {
          navigator.clipboard.writeText(activeArtifact.code);
      }
  }

  // Helper to remove code blocks from chat text so they don't double render
  const cleanChatText = (text: string) => {
      return text.replace(/```[\s\S]*?(?:```|$)/g, "*[Code rendering in Genix Workspace...]*");
  }

  return (
    <div className="flex-1 flex flex-row overflow-hidden w-full max-w-full h-full bg-background relative">
      
      {/* Left Chat Pane */}
      <div className={`flex flex-col relative h-full overflow-hidden transition-all duration-500 ease-in-out ${activeArtifact ? 'w-1/2 border-r border-border' : 'w-full items-center'}`}>
        
        {messages.length === 0 ? (
          // Empty State
          <div className="w-full max-w-4xl flex flex-col items-center justify-center flex-1 h-full px-4">
            <h1 ref={titleRef} className="text-4xl md:text-5xl font-serif italic font-medium text-foreground mb-12 text-center">
              AgenticML Workspace
            </h1>
            <InputForm 
              input={input} 
              setInput={setInput} 
              handleSubmit={handleSubmit} 
              isProcessing={isProcessing} 
              containerRef={inputContainerRef} 
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
            />
          </div>
        ) : (
          // Active Chat State
          <div className={`w-full flex flex-col h-full pt-8 pb-4 px-4 relative ${activeArtifact ? 'max-w-full' : 'max-w-4xl'}`}>
            <div ref={messagesContainerRef} className="flex-1 overflow-y-auto scrollbar-hide pb-32 flex flex-col gap-6 px-2 w-full">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex flex-col w-full ${msg.agent === 'User' ? 'items-end' : 'items-start'}`}>
                  {msg.agent !== 'User' && (
                    <span className="text-xs font-semibold text-primary/80 uppercase tracking-wider mb-2 flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4" />
                      {msg.agent}
                    </span>
                  )}
                  <div className={`p-5 rounded-2xl w-fit max-w-[95%] text-[15px] leading-relaxed shadow-sm break-words ${
                    msg.agent === 'User' 
                      ? 'bg-foreground text-background rounded-tr-sm' 
                      : 'glass-panel bg-white/80 border border-border rounded-tl-sm text-foreground'
                  }`}>
                    {msg.agent === 'User' ? msg.message : (
                        <div className="prose prose-sm max-w-none dark:prose-invert">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {cleanChatText(msg.message)}
                            </ReactMarkdown>
                        </div>
                    )}
                  </div>
                </div>
              ))}
              {isProcessing && (
                <div className="flex items-center gap-3 text-sm text-muted-foreground italic pl-2 py-4">
                  <CircleDashed className="w-5 h-5 animate-spin text-primary" />
                  <span className="animate-pulse">Genix is orchestrating pipeline...</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className={`absolute bottom-6 left-1/2 -translate-x-1/2 w-full px-4 bg-gradient-to-t from-background via-background/90 to-transparent pt-6 pb-2 ${activeArtifact ? 'max-w-full' : 'max-w-4xl'}`}>
              <InputForm 
                input={input} 
                setInput={setInput} 
                handleSubmit={handleSubmit} 
                isProcessing={isProcessing} 
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
              />
            </div>
          </div>
        )}
      </div>

      {/* Right Artifact Pane */}
      {activeArtifact && (
          <div ref={artifactPaneRef} className="w-1/2 h-full flex flex-col bg-[#1e1e1e] text-white shadow-2xl relative z-10 overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 bg-[#2d2d2d] border-b border-[#404040]">
                  <div className="flex items-center gap-2">
                      <FileCode2 className="w-4 h-4 text-primary" />
                      <span className="text-sm font-medium font-mono text-gray-200">generated_pipeline.{activeArtifact.language === 'python' ? 'py' : 'txt'}</span>
                  </div>
                  <div className="flex gap-2">
                      <button onClick={copyToClipboard} className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#404040] hover:bg-[#505050] text-xs font-medium text-gray-200 transition-colors">
                          <Copy className="w-3.5 h-3.5" />
                          Copy Code
                      </button>
                  </div>
              </div>

              {/* Code Editor Body */}
              <div className="flex-1 overflow-auto bg-[#1e1e1e] scrollbar-thin scrollbar-thumb-[#404040]">
                  <SyntaxHighlighter
                      language={activeArtifact.language || "python"}
                      style={vscDarkPlus}
                      customStyle={{ margin: 0, padding: '1.5rem', background: 'transparent', fontSize: '14px' }}
                      showLineNumbers={true}
                      wrapLines={true}
                  >
                      {activeArtifact.code}
                  </SyntaxHighlighter>
              </div>
          </div>
      )}

    </div>
  );
interface InputFormProps {
  input: string;
  setInput: (value: string) => void;
  handleSubmit: (e: React.FormEvent) => void;
  isProcessing: boolean;
  containerRef: React.RefObject<HTMLDivElement | null>;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
}

function InputForm({ input, setInput, handleSubmit, isProcessing, containerRef, selectedModel, setSelectedModel }: InputFormProps) {
  return (
    <div ref={containerRef} className="w-full shadow-lg rounded-2xl bg-white/70 backdrop-blur-xl border border-border">
      <form 
        onSubmit={handleSubmit}
        className="w-full rounded-2xl focus-within:shadow-md focus-within:ring-1 focus-within:ring-primary/50 transition-all duration-300 flex flex-col overflow-hidden"
      >
        <textarea 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type / for skills or describe your ML workflow..."
          className="w-full min-h-[80px] max-h-[250px] p-4 bg-transparent outline-none resize-none text-foreground placeholder:text-muted-foreground font-light text-base"
          autoFocus
          disabled={isProcessing}
          suppressHydrationWarning
          onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
              }
          }}
        />

        <div className="p-3 flex justify-between items-center border-t border-border/30 bg-secondary/10">
          <div className="flex gap-2 text-muted-foreground">
            <button type="button" className="p-2 hover:bg-black/5 hover:text-foreground rounded-lg transition-colors">
              <Paperclip className="w-4 h-4" />
            </button>
          </div>

          <div className="flex gap-3 items-center">
            <select 
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="text-xs text-secondary-foreground font-medium hidden sm:inline-block px-2 py-1 bg-black/5 hover:bg-black/10 rounded-md outline-none cursor-pointer border-none transition-colors"
            >
              <option value="Gemini 3.5 Flash">Gemini 3.5 Flash</option>
              <option value="Gemini 3.1 Pro">Gemini 3.1 Pro</option>
              <option value="Groq Llama 3.1">Groq Llama 3.1</option>
              <option value="Groq Mixtral">Groq Mixtral</option>
            </select>
            <button 
              type="submit" 
              disabled={!input.trim() || isProcessing}
              className="p-2.5 bg-primary text-primary-foreground rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-all shadow-sm group"
            >
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
