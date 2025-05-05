'use client';

import Image from 'next/image';
import { type ChangeEvent, useState, type FC } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload } from 'lucide-react';

// Inline SVG for MRI Icon (as Lucide doesn't have a direct one)
const MriIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-10 w-10 text-muted-foreground">
        <path d="M12 2a10 10 0 1 0 10 10" />
        <path d="M12 2a10 10 0 0 0-2.2 19.6" />
        <path d="M12 2a10 10 0 0 1 7.8 3.2" />
        <path d="M12 22a10 10 0 0 0 2.2-19.6" />
        <path d="M12 22a10 10 0 0 1-7.8-3.2" />
        <path d="M7.1 4.1a10 10 0 0 0-3 3" />
        <path d="M4.1 7.1a10 10 0 0 0-.1 9.8" />
        <path d="M7.1 19.9a10 10 0 0 0 3 .1" />
        <path d="M10.1 21.9a10 10 0 0 0 9.8 .1" />
        <path d="M16.9 19.9a10 10 0 0 0 3-3" />
        <path d="M19.9 16.9a10 10 0 0 0 .1-9.8" />
        <path d="M16.9 4.1a10 10 0 0 0-3-.1" />
        <path d="M13.9 2.1a10 10 0 0 0-9.8-.1" />
        <path d="M12 5a7 7 0 1 0 7 7" />
    </svg>
);


interface MriUploaderProps {
  onImageUpload: (imageDataUri: string | null) => void;
  isGenerating: boolean;
}

const MriUploader: FC<MriUploaderProps> = ({ onImageUpload, isGenerating }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setPreview(result);
        setFileName(file.name);
        onImageUpload(result);
      };
      reader.readAsDataURL(file);
    } else {
      setPreview(null);
      setFileName(null);
      onImageUpload(null);
    }
  };

  const handleClear = () => {
    setPreview(null);
    setFileName(null);
    onImageUpload(null);
    // Reset the input field value
    const input = document.getElementById('mri-upload') as HTMLInputElement;
    if (input) {
      input.value = '';
    }
  };

  return (
    <Card className="w-full shadow-md">
      <CardHeader>
        <CardTitle>Upload MRI Scan</CardTitle>
        <CardDescription>Select a PNG or JPEG image of the MRI scan.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <Label htmlFor="mri-upload">MRI Image</Label>
          <Input
            id="mri-upload"
            type="file"
            accept="image/png, image/jpeg"
            onChange={handleFileChange}
            disabled={isGenerating}
            className="file:text-foreground"
          />
        </div>
        {preview && (
          <div className="mt-4 space-y-2 rounded-lg border bg-secondary p-4 shadow-inner">
            <p className="text-sm font-medium text-secondary-foreground">Image Preview:</p>
            <Image
              src={preview}
              alt="MRI Scan Preview"
              width={200}
              height={200}
              className="mx-auto rounded-md border object-contain"
              data-ai-hint="mri scan brain"
            />
            <p className="truncate text-xs text-muted-foreground">File: {fileName}</p>
          </div>
        )}
         {!preview && (
          <div className="mt-4 flex flex-col items-center justify-center space-y-2 rounded-lg border border-dashed bg-muted p-8 text-center">
            <MriIcon />
            <p className="text-sm text-muted-foreground">No image selected</p>
          </div>
        )}
      </CardContent>
       {preview && (
        <CardFooter>
          <Button variant="outline" onClick={handleClear} disabled={isGenerating}>
            Clear Image
          </Button>
        </CardFooter>
      )}
    </Card>
  );
};

export default MriUploader;
