"use client";

import { useChat } from "../layout";
import { Search } from "lucide-react";

export default function HistoryPage() {
  const { history } = useChat();

  // Mocked older history to match the reference image
  const olderHistory = [
    { label: "Building an agentic ML model developer platform", time: "6 hours ago", author: "Agentic" },
    { label: "Agentic AI project ideas for final year", time: "8 hours ago", author: "" },
    { label: "Final year project group change request", time: "15 hours ago", author: "" },
    { label: "LLM fundamentals and prompt engineering roadmap", time: "Jun 19", author: "Agentic" },
    { label: "Weight initialization in deep learning", time: "Jun 17", author: "" },
    { label: "Python frameworks", time: "Jun 17", author: "" },
    { label: "Quick dabs guide for interview prep", time: "Jun 16", author: "" },
    { label: "Mastering complex agentic AI across platforms", time: "Jun 11", author: "Agentic" },
    { label: "Why not use auto instead of explicit type declarations", time: "Jun 9", author: "" },
  ];

  return (
    <div className="flex flex-col h-full w-full bg-background p-8 overflow-y-auto">
      <div className="max-w-4xl w-full mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-serif text-foreground">Chats</h1>
          <div className="flex gap-3">
            <select className="bg-card/50 border border-border rounded-lg px-4 py-2 text-sm text-foreground outline-none">
              <option>Filter by All</option>
            </select>
            <button className="bg-card/50 border border-border hover:bg-black/5 text-foreground px-4 py-2 rounded-lg text-sm transition-colors">
              Select chats
            </button>
            <button className="bg-primary text-primary-foreground px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors">
              New chat
            </button>
          </div>
        </div>

        <div className="relative mb-8 group">
          <Search className="w-5 h-5 absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground group-focus-within:text-primary transition-colors" />
          <input 
            type="text" 
            placeholder="Search chats..." 
            className="w-full bg-card/30 border border-border rounded-xl py-3 pl-12 pr-4 text-foreground placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-primary/50 transition-all shadow-sm"
          />
        </div>

        <div className="flex flex-col divide-y divide-border/50">
          {history.map((label, idx) => (
            <div key={`new-${idx}`} className="flex items-center justify-between py-4 group hover:bg-black/5 px-2 rounded-lg transition-colors cursor-pointer -mx-2">
              <span className="text-[15px] text-foreground font-medium">{label}</span>
              <span className="text-sm text-muted-foreground">Just now</span>
            </div>
          ))}
          
          {olderHistory.map((item, idx) => (
            <div key={`old-${idx}`} className="flex items-center justify-between py-4 group hover:bg-black/5 px-2 rounded-lg transition-colors cursor-pointer -mx-2">
              <div className="flex items-center gap-3">
                <span className="text-[15px] text-foreground hover:underline decoration-muted-foreground underline-offset-4">{item.label}</span>
              </div>
              <div className="flex items-center gap-6">
                {item.author && <span className="text-xs text-muted-foreground font-medium">{item.author}</span>}
                <span className="text-sm text-muted-foreground w-24 text-right">{item.time}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
