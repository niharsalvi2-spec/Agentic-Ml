"use client";

import { Briefcase, AppWindow, FileBadge, Plug } from "lucide-react";

export default function CustomizePage() {
  return (
    <div className="flex flex-col h-full w-full bg-background overflow-y-auto items-center justify-center p-8">
      <div className="max-w-2xl w-full flex flex-col items-center">
        
        {/* Header Section */}
        <div className="flex flex-col items-center mb-10">
          <Briefcase className="w-16 h-16 text-foreground mb-6" strokeWidth={1.5} />
          <h1 className="text-3xl font-serif text-foreground mb-3 text-center">
            Customize AgenticML
          </h1>
          <p className="text-muted-foreground text-center">
            Skills, connectors, and plugins shape how AgenticML works with you.
          </p>
        </div>

        {/* Options List */}
        <div className="w-full flex flex-col gap-4">
          
          <CustomizeCard 
            icon={<AppWindow className="w-5 h-5 text-foreground" />}
            title="Connect your apps"
            description="Let AgenticML read and write to the tools you already use."
          />
          
          <CustomizeCard 
            icon={<FileBadge className="w-5 h-5 text-foreground" />}
            title="Create new skills"
            description="Teach AgenticML your processes, team norms, and expertise."
          />
          
          <CustomizeCard 
            icon={<Plug className="w-5 h-5 text-foreground" />}
            title="Browse plugins"
            description="Add pre-built knowledge for your field."
          />

        </div>
      </div>
    </div>
  );
}

function CustomizeCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="flex items-center gap-5 p-5 rounded-2xl border border-border bg-card/30 hover:bg-black/5 hover:border-border/80 cursor-pointer transition-all group w-full">
      <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-background border border-border group-hover:scale-105 transition-transform shadow-sm">
        {icon}
      </div>
      <div className="flex flex-col">
        <span className="text-[15px] font-medium text-foreground mb-0.5">{title}</span>
        <span className="text-sm text-muted-foreground">{description}</span>
      </div>
    </div>
  );
}
