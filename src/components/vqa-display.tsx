'use client';

import { type FC } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { MessageCircleQuestion } from 'lucide-react'; // Use appropriate icon

interface VqaDisplayProps {
  answer: string | null;
  isGenerating: boolean;
}

const VqaDisplay: FC<VqaDisplayProps> = ({ answer, isGenerating }) => {
  return (
    <Card className="w-full shadow-md">
      <CardHeader>
        <div className="flex items-center gap-2">
           <MessageCircleQuestion className="h-5 w-5 text-primary" />
           <CardTitle>VQA Answer</CardTitle>
        </div>
        <CardDescription>AI-generated answer to your question about the MRI scan.</CardDescription>
      </CardHeader>
      <CardContent>
        {isGenerating ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        ) : answer ? (
          <p className="whitespace-pre-wrap rounded-md border bg-secondary p-4 text-sm text-secondary-foreground shadow-inner">
            {answer}
          </p>
        ) : (
          <p className="text-center text-sm text-muted-foreground">Ask a question about the uploaded MRI scan to see the answer here.</p>
        )}
      </CardContent>
    </Card>
  );
};

export default VqaDisplay;
