'use client';

import { generateMriReport } from '@/ai/flows/generate-mri-report';
import Header from '@/components/layout/header';
import MriUploader from '@/components/mri-uploader';
import ReportDisplay from '@/components/report-display';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useState } from 'react';
import { Loader2, AlertTriangle } from 'lucide-react';

export default function Home() {
  const [mriImageDataUri, setMriImageDataUri] = useState<string | null>(null);
  const [clinicalSummary, setClinicalSummary] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (imageDataUri: string | null) => {
    setMriImageDataUri(imageDataUri);
    setClinicalSummary(null); // Clear previous report when image changes
    setError(null); // Clear previous errors
  };

  const handleGenerateReport = async () => {
    if (!mriImageDataUri) {
      setError("Please upload an MRI image first.");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setClinicalSummary(null); // Clear previous report before generating new one

    try {
      const result = await generateMriReport({ mriImageDataUri });
      setClinicalSummary(result.clinicalSummary);
    } catch (err) {
      console.error("Error generating report:", err);
      setError("Failed to generate report. Please ensure the image is valid and try again.");
      setClinicalSummary(null);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-secondary">
      <Header />
      <main className="container mx-auto flex flex-1 flex-col items-center gap-8 p-4 py-8 md:p-6 lg:p-8">
        <div className="grid w-full max-w-4xl grid-cols-1 gap-6 md:grid-cols-2">
          <MriUploader onImageUpload={handleImageUpload} isGenerating={isGenerating} />
          <ReportDisplay report={clinicalSummary} isGenerating={isGenerating} />
        </div>

        {error && (
           <Alert variant="destructive" className="w-full max-w-4xl">
             <AlertTriangle className="h-4 w-4" />
             <AlertTitle>Error</AlertTitle>
             <AlertDescription>{error}</AlertDescription>
           </Alert>
         )}

        <Button
          onClick={handleGenerateReport}
          disabled={!mriImageDataUri || isGenerating}
          size="lg"
          className="w-full max-w-xs shadow-md"
        >
          {isGenerating ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            'Generate Report'
          )}
        </Button>


      </main>
      <footer className="py-4 text-center text-sm text-muted-foreground">
         NeuroReport © {new Date().getFullYear()}
      </footer>
    </div>
  );
}
