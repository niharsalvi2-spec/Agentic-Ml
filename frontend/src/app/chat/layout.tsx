"use client";

import { ReactNode, useState, createContext, useContext, useEffect } from "react";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { Plus, MessageSquare, Folder, Code, Settings, LogOut } from "lucide-react";
import { useSession, signOut } from "next-auth/react";

interface ChatContextType {
  history: string[];
  addHistory: (prompt: string) => void;
}

const ChatContext = createContext<ChatContextType>({ history: [], addHistory: () => {} });

export function useChat() {
  return useContext(ChatContext);
}

export default function ChatLayout({ children }: { children: ReactNode }) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const pathname = usePathname();
  
  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login");
    }
  }, [status, router]);
  
  const [history, setHistory] = useState<string[]>([
    "Churn prediction model...",
    "Fine-tune Llama 3 on...",
    "Vision transformer for..."
  ]);

  const addHistory = (prompt: string) => {
    const label = prompt.length > 25 ? prompt.substring(0, 25) + "..." : prompt;
    setHistory(prev => [label, ...prev]);
  };

  return (
    <ChatContext.Provider value={{ history, addHistory }}>
      <div className="flex h-screen bg-background overflow-hidden font-sans">
        {/* Sidebar */}
        <aside className="w-64 border-r border-border bg-card/30 flex flex-col justify-between hidden md:flex">
          <div className="p-4 flex flex-col gap-6 h-full overflow-hidden">
            <div className="flex items-center justify-between">
              <Link href="/" className="text-xl font-serif italic font-bold tracking-tight text-foreground">
                Agentic<span className="text-primary not-italic font-sans font-light">ML</span>
              </Link>
            </div>

            <Link 
              href="/chat"
              className="flex items-center gap-2 bg-primary text-primary-foreground px-4 py-2 rounded-full font-medium hover:bg-primary/90 transition-colors shadow-sm w-full justify-center text-sm"
            >
              <Plus className="w-4 h-4" />
              New Chat
            </Link>

            <nav className="flex flex-col gap-1 mt-4">
              <NavItem href="/chat/history" icon={<MessageSquare className="w-4 h-4" />} label="Chats" active={pathname === "/chat/history"} />
              <NavItem href="/chat/projects" icon={<Folder className="w-4 h-4" />} label="Projects" active={pathname === "/chat/projects"} />
              <NavItem href="#" icon={<Code className="w-4 h-4" />} label="Code" active={false} onClick={() => alert("Code feature coming soon!")} />
              <NavItem href="/chat/customize" icon={<Settings className="w-4 h-4" />} label="Customize" active={pathname === "/chat/customize"} />
            </nav>

            <div className="mt-6 flex-1 overflow-hidden flex flex-col">
              <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 px-2">Recents</h3>
              <div className="flex flex-col gap-1 overflow-y-auto pr-2 scrollbar-hide pb-4">
                {history.map((item, idx) => (
                  <RecentItem key={idx} label={item} />
                ))}
              </div>
            </div>
          </div>

          <div className="p-4 border-t border-border">
            <div className="flex items-center justify-between hover:bg-black/5 p-2 rounded-xl transition-colors group">
              <div className="flex items-center gap-3 cursor-pointer">
                <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-sm uppercase">
                  {session?.user?.name ? session.user.name[0] : "U"}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-medium text-foreground truncate max-w-[100px]">
                    {session?.user?.name || "Guest User"}
                  </span>
                  <span className="text-xs text-muted-foreground">Free Plan</span>
                </div>
              </div>
              <button onClick={() => signOut({ callbackUrl: "/" })} className="text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity p-1">
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col relative h-screen w-full min-w-0">
          {children}
        </main>
      </div>
    </ChatContext.Provider>
  );
}

function NavItem({ icon, label, href, active = false, onClick }: { icon: React.ReactNode, label: string, href: string, active?: boolean, onClick?: () => void }) {
  if (onClick) {
    return (
      <div 
        onClick={onClick}
        className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors text-sm font-medium ${
          active ? "bg-primary/10 text-primary" : "text-secondary-foreground hover:bg-black/5 hover:text-foreground"
        }`}
      >
        {icon}
        <span>{label}</span>
      </div>
    );
  }
  return (
    <Link 
      href={href}
      className={`flex items-center gap-3 px-3 py-2 rounded-lg transition-colors text-sm font-medium ${
        active ? "bg-primary/10 text-primary" : "text-secondary-foreground hover:bg-black/5 hover:text-foreground"
      }`}
    >
      {icon}
      <span>{label}</span>
    </Link>
  );
}

function RecentItem({ label }: { label: string }) {
  return (
    <div className="truncate px-3 py-1.5 text-sm text-secondary-foreground hover:bg-black/5 hover:text-foreground rounded-lg cursor-pointer transition-colors">
      {label}
    </div>
  );
}
