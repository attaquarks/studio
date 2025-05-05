import { Brain } from 'lucide-react';
import { type FC } from 'react';

interface HeaderProps {}

const Header: FC<HeaderProps> = ({}) => {
  return (
    <header className="border-b bg-card">
      <div className="container mx-auto flex h-16 items-center justify-between px-4 md:px-6">
        <div className="flex items-center gap-2">
          <Brain className="h-6 w-6 text-primary" />
          <h1 className="text-xl font-semibold text-foreground">NeuroReport</h1>
        </div>
        {/* Add navigation or user controls here if needed */}
      </div>
    </header>
  );
};

export default Header;
