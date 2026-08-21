"use client";

import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import Scene from "@/components/Scene";
import { ArrowRight, Sparkles } from "lucide-react";

export default function Home() {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const subRef = useRef<HTMLParagraphElement>(null);
  const promptRef = useRef<HTMLDivElement>(null);
  const [prompt, setPrompt] = useState("");

  useEffect(() => {
    const tl = gsap.timeline();
    
    tl.fromTo(headingRef.current, 
      { y: 40, opacity: 0, filter: "blur(10px)" }, 
      { y: 0, opacity: 1, filter: "blur(0px)", duration: 1.2, ease: "power4.out", delay: 0.2 }
    )
    .fromTo(subRef.current, 
      { y: 20, opacity: 0 }, 
      { y: 0, opacity: 1, duration: 1, ease: "power3.out" },
      "-=0.8"
    )
    .fromTo(promptRef.current, 
      { y: 20, opacity: 0, scale: 0.95 }, 
      { y: 0, opacity: 1, scale: 1, duration: 1, ease: "expo.out" },
      "-=0.6"
    );
  }, []);

  const handlePromptSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      window.location.href = "/pipeline";
      return;
    }
    window.location.href = `/pipeline?prompt=${encodeURIComponent(prompt)}`;
  };

  return (
    <main className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-background">
      {/* 3D Background */}
      <Scene />
      
      {/* Content Overlay */}
      <div className="z-10 w-full px-6 max-w-5xl mx-auto mt-[5vh] flex flex-col items-center">
        
        {/* Subtle glass backing behind text to ensure perfect contrast against 3D element */}
        <div className="glass-panel p-10 md:p-14 rounded-[2rem] text-center w-full max-w-4xl relative overflow-hidden">
          {/* Decorative glare */}
          <div className="absolute -top-24 -left-24 w-48 h-48 bg-primary/20 rounded-full blur-[50px] pointer-events-none" />
          
          <h1 
            ref={headingRef} 
            className="text-5xl md:text-7xl lg:text-8xl font-light tracking-tight text-foreground mb-6"
          >
            Design your <br />
            <span className="italic font-serif text-primary font-medium tracking-normal">ML Workflows</span>
          </h1>
          
          <p 
            ref={subRef} 
            className="text-lg md:text-2xl text-foreground/80 mb-10 max-w-2xl mx-auto font-light leading-relaxed"
          >
            Simply describe what you want to build. Our Agentic AI will orchestrate, train, and deploy your models instantly.
          </p>
          
          <div ref={promptRef} className="w-full max-w-2xl mx-auto">
            <form 
              onSubmit={handlePromptSubmit} 
              className="relative flex items-center w-full bg-white/80 border border-border shadow-lg rounded-full overflow-hidden focus-within:ring-2 focus-within:ring-primary/50 focus-within:border-primary transition-all duration-300"
            >
              <div className="pl-6 text-primary">
                <Sparkles className="w-6 h-6 opacity-70" />
              </div>
              <input 
                type="text" 
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="E.g., Build a churn prediction model using customer data..." 
                className="w-full px-4 py-6 bg-transparent text-foreground placeholder:text-foreground/40 font-light text-lg outline-none"
                suppressHydrationWarning
              />
              <button 
                type="submit" 
                className="absolute right-3 bg-primary text-primary-foreground p-3 rounded-full hover:scale-105 transition-transform duration-300 shadow-md flex items-center justify-center group"
                suppressHydrationWarning
              >
                <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
              </button>
            </form>
            
            <div className="mt-6 flex flex-wrap justify-center gap-3 text-sm font-medium text-foreground/60">
              <span className="cursor-pointer hover:text-primary transition-colors bg-black/5 px-3 py-1 rounded-full">Vision Models</span>
              <span className="cursor-pointer hover:text-primary transition-colors bg-black/5 px-3 py-1 rounded-full">NLP Tasks</span>
              <span className="cursor-pointer hover:text-primary transition-colors bg-black/5 px-3 py-1 rounded-full">Tabular Analytics</span>
            </div>
          </div>
        </div>

      </div>
    </main>
  );
}
