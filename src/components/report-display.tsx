import { type FC } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { FileText } from 'lucide-react';

interface ReportDisplayProps {
  report: string | null;
  isGenerating: boolean;
}

const ReportDisplay: FC<ReportDisplayProps> = ({ report, isGenerating }) => {
  return (
    <Card className="w-full shadow-md">
      <CardHeader>
        <div className="flex items-center gap-2">
           <FileText className="h-5 w-5 text-primary" />
           <CardTitle>Clinical Summary</CardTitle>
        </div>
        <CardDescription>AI-generated summary based on the uploaded MRI scan.</CardDescription>
      </CardHeader>
      <CardContent>
        {isGenerating ? (
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        ) : report ? (
          <p className="whitespace-pre-wrap rounded-md border bg-secondary p-4 text-sm text-secondary-foreground shadow-inner">
            {report}
          </p>
        ) : (
          <p className="text-center text-sm text-muted-foreground">Upload an MRI image and click "Generate Report" to see the summary here.</p>
        )}
      </CardContent>
    </Card>
  );
};

export default ReportDisplay;
