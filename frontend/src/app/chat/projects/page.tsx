"use client";

import { Search } from "lucide-react";

export default function ProjectsPage() {
  const projects = [
    { title: "Agentic ML Platform", updated: "Updated just now" },
    { title: "Agentic Ethical Hacker", updated: "Updated May 17" },
  ];

  return (
    <div className="flex flex-col h-full w-full bg-background p-8 overflow-y-auto">
      <div className="max-w-4xl w-full mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-serif text-foreground">Projects</h1>
          <div className="flex gap-3">
            <select className="bg-card/50 border border-border rounded-lg px-4 py-2 text-sm text-foreground outline-none">
              <option>Sort by Last updated</option>
            </select>
            <button className="bg-primary text-primary-foreground px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary/90 transition-colors shadow-sm">
              New project
            </button>
          </div>
        </div>

        <div className="relative mb-8 group">
          <Search className="w-5 h-5 absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground group-focus-within:text-primary transition-colors" />
          <input 
            type="text" 
            placeholder="Search projects..." 
            className="w-full bg-card/30 border border-border rounded-xl py-3 pl-12 pr-4 text-foreground placeholder:text-muted-foreground outline-none focus:ring-1 focus:ring-primary/50 transition-all shadow-sm"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {projects.map((project, idx) => (
            <div 
              key={idx} 
              className="glass-panel bg-white/50 border border-border rounded-2xl p-6 hover:shadow-md transition-all cursor-pointer group flex flex-col justify-between h-40"
            >
              <h3 className="text-lg font-medium text-foreground group-hover:text-primary transition-colors">
                {project.title}
              </h3>
              <p className="text-sm text-muted-foreground">
                {project.updated}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
