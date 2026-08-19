"use client";

import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { signIn } from "next-auth/react";

export default function LoginPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  useEffect(() => {
    gsap.fromTo(containerRef.current, 
      { opacity: 0, y: 30, filter: "blur(10px)" }, 
      { opacity: 1, y: 0, filter: "blur(0px)", duration: 1, ease: "power4.out" }
    );
  }, []);

  const handleEmailLogin = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("Logging in with:", email);
  };

  const handleGoogleLogin = () => {
    signIn("google", { callbackUrl: "/chat" });
  };

  return (
    <div className="min-h-screen bg-background flex flex-col justify-center items-center relative overflow-hidden px-4">
      {/* Decorative Orbs */}
      <div className="absolute top-1/3 left-1/4 w-72 h-72 bg-primary/20 rounded-full blur-[80px] pointer-events-none" />
      <div className="absolute bottom-1/3 right-1/4 w-72 h-72 bg-secondary-foreground/10 rounded-full blur-[80px] pointer-events-none" />

      <div ref={containerRef} className="glass-panel p-10 md:p-12 rounded-[2rem] w-full max-w-md relative z-10">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-light tracking-tight text-foreground mb-2">Welcome Back</h1>
          <p className="text-secondary-foreground font-light text-sm">Sign in to your AgenticML account</p>
        </div>

        <button 
          onClick={handleGoogleLogin}
          className="w-full flex items-center justify-center gap-3 bg-white/60 border border-border text-foreground py-3 px-4 rounded-full font-medium hover:bg-white hover:shadow-md transition-all duration-300 mb-6"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/><path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/><path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/><path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/></svg>
          Continue with Google
        </button>

        <div className="flex items-center gap-4 mb-6">
          <div className="h-[1px] flex-1 bg-border" />
          <span className="text-xs text-muted-foreground uppercase tracking-widest">Or</span>
          <div className="h-[1px] flex-1 bg-border" />
        </div>

        <form onSubmit={handleEmailLogin} className="space-y-4">
          <div>
            <input 
              type="email" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email address"
              className="w-full px-4 py-3 bg-white/40 border border-border rounded-xl focus:ring-2 focus:ring-primary/50 focus:border-primary outline-none transition-all placeholder:text-muted-foreground font-light"
              required
            />
          </div>
          <div>
            <input 
              type="password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              className="w-full px-4 py-3 bg-white/40 border border-border rounded-xl focus:ring-2 focus:ring-primary/50 focus:border-primary outline-none transition-all placeholder:text-muted-foreground font-light"
              required
            />
          </div>
          
          <div className="flex justify-between items-center px-1">
            <label className="flex items-center gap-2 text-sm text-secondary-foreground font-light cursor-pointer">
              <input type="checkbox" className="rounded border-border text-primary focus:ring-primary" />
              Remember me
            </label>
            <Link href="/forgot-password" className="text-sm text-primary hover:underline font-light">
              Forgot password?
            </Link>
          </div>

          <button 
            type="submit" 
            className="w-full flex items-center justify-center gap-2 bg-primary text-primary-foreground py-3.5 px-4 rounded-xl font-medium hover:bg-primary/90 hover:shadow-lg transition-all duration-300 group mt-4"
          >
            Sign In
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </button>
        </form>

        <p className="text-center text-sm text-secondary-foreground mt-8 font-light">
          Don't have an account?{" "}
          <Link href="/signup" className="text-primary font-medium hover:underline">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}
