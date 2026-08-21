"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";

export default function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  if (pathname.startsWith("/chat")) return null;

  return (
    <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${
      scrolled ? "bg-background/80 backdrop-blur-md border-b border-border py-4" : "bg-transparent py-6"
    }`}>
      <div className="max-w-7xl mx-auto px-6 flex justify-between items-center">
        <Link href="/" className="text-2xl font-serif italic font-bold tracking-tight text-foreground">
          Agentic<span className="text-primary not-italic font-sans font-light">ML</span>
        </Link>
        
        <div className="hidden md:flex gap-8 items-center">
          <Link 
            href="/" 
            className={`text-sm font-medium uppercase tracking-widest transition-colors ${
              pathname === "/" ? "text-primary" : "text-foreground hover:text-primary"
            }`}
          >
            Home
          </Link>
          <Link 
            href="/pipeline" 
            className={`text-sm font-medium uppercase tracking-widest transition-colors ${
              pathname === "/pipeline" ? "text-primary font-bold" : "text-foreground hover:text-primary"
            }`}
          >
            Pipeline Studio
          </Link>
          <Link 
            href="/pricing" 
            className={`text-sm font-medium uppercase tracking-widest transition-colors ${
              pathname === "/pricing" ? "text-primary" : "text-foreground hover:text-primary"
            }`}
          >
            Pricing
          </Link>
          <Link 
            href="/docs" 
            className={`text-sm font-medium uppercase tracking-widest transition-colors ${
              pathname === "/docs" ? "text-primary" : "text-foreground hover:text-primary"
            }`}
          >
            Docs
          </Link>
        </div>

        <div className="flex gap-4">
          <Link href="/login" className="text-sm font-medium uppercase tracking-widest text-foreground hover:text-primary px-4 py-2 transition-colors">
            Login
          </Link>
          <Link href="/signup" className="text-sm font-medium uppercase tracking-widest bg-primary text-primary-foreground px-6 py-2 rounded-full hover:bg-primary/90 transition-colors shadow-md">
            Get Started
          </Link>
        </div>
      </div>
    </nav>
  );
}
