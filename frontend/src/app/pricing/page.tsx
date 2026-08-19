"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import Link from "next/link";

const plans = [
  {
    name: "Personal",
    price: "$0",
    description: "For hobbyists and students exploring AI.",
    features: ["1 Active Project", "Standard Compute", "Community Support", "Basic Analytics"],
    buttonText: "Begin Journey",
    highlighted: false,
  },
  {
    name: "Professional",
    price: "$49",
    period: "/mo",
    description: "For professional teams scaling their infrastructure.",
    features: ["Unlimited Projects", "GPU Acceleration", "Priority Support", "Advanced XAI Tools", "Custom LangGraph Workflows"],
    buttonText: "Upgrade to Pro",
    highlighted: true,
  },
  {
    name: "Enterprise",
    price: "Custom",
    description: "For global organizations requiring maximum power.",
    features: ["Dedicated Infrastructure", "Custom SLA", "24/7 Phone Support", "On-Premise Deployment"],
    buttonText: "Contact Sales",
    highlighted: false,
  }
];

export default function PricingPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cardsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    gsap.fromTo(containerRef.current, 
      { opacity: 0, y: 40, filter: "blur(5px)" }, 
      { opacity: 1, y: 0, filter: "blur(0px)", duration: 1.2, ease: "power4.out" }
    );

    gsap.fromTo(cardsRef.current,
      { opacity: 0, y: 60, scale: 0.95 },
      { opacity: 1, y: 0, scale: 1, duration: 1.2, stagger: 0.2, ease: "expo.out", delay: 0.3 }
    );
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground py-32 px-4 sm:px-6 lg:px-8 overflow-hidden relative">
      {/* Decorative Blur Backgrounds */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-primary/10 rounded-full blur-[100px] pointer-events-none" />

      <div ref={containerRef} className="max-w-7xl mx-auto text-center mb-24 relative z-10">
        <h1 className="text-5xl md:text-7xl font-light tracking-tight text-foreground mb-6 font-serif italic">
          Transparent Pricing
        </h1>
        <p className="text-xl text-secondary-foreground max-w-2xl mx-auto font-light">
          Elevate your workflow with plans designed to scale with your ambition.
        </p>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-10 relative z-10">
        {plans.map((plan, i) => (
          <div 
            key={plan.name}
            ref={el => { cardsRef.current[i] = el; }}
            className={`relative rounded-3xl p-10 flex flex-col transition-all duration-500 hover:-translate-y-2 ${
              plan.highlighted 
                ? "glass-panel ring-1 ring-primary/30" 
                : "bg-white/40 backdrop-blur-md border border-border"
            }`}
          >
            {plan.highlighted && (
              <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                <span className="bg-primary text-primary-foreground text-xs font-bold tracking-widest uppercase px-6 py-2 rounded-full shadow-lg">
                  Most Popular
                </span>
              </div>
            )}
            
            <div className="mb-8">
              <h2 className="text-3xl font-light tracking-tight text-foreground mb-3">{plan.name}</h2>
              <p className="text-secondary-foreground font-light leading-relaxed">{plan.description}</p>
            </div>
            
            <div className="mb-10 flex items-baseline font-serif">
              <span className="text-6xl font-light text-foreground tracking-tighter">{plan.price}</span>
              {plan.period && <span className="text-xl text-secondary-foreground ml-2">{plan.period}</span>}
            </div>

            <ul className="flex-1 space-y-6 mb-12">
              {plan.features.map((feature) => (
                <li key={feature} className="flex items-center text-secondary-foreground font-light tracking-wide">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary mr-4 opacity-70" />
                  {feature}
                </li>
              ))}
            </ul>

            <Link 
              href={plan.name === "Enterprise" ? "/contact" : "/signup"}
              className={`w-full py-4 px-6 rounded-full font-medium text-center tracking-wide uppercase text-sm transition-all duration-300 ${
                plan.highlighted
                  ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg hover:shadow-xl"
                  : "bg-transparent border border-foreground/20 text-foreground hover:bg-foreground/5"
              }`}
            >
              {plan.buttonText}
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}
