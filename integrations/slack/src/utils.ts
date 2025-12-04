/**
 * Utility functions for the LazAI Slack Bot
 */

export function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

export function validateFileType(filename: string): boolean {
    const allowedExtensions = ['.txt', '.json', '.csv', '.jsonl', '.parquet', '.md'];
    const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));
    return allowedExtensions.includes(extension);
}

export function sanitizeFilename(filename: string): string {
    // Remove or replace invalid characters
    return filename.replace(/[^a-z0-9._-]/gi, '_').toLowerCase();
}

export function generateProgressBar(current: number, total: number, length: number = 20): string {
    const progress = Math.round((current / total) * length);
    const filled = '█'.repeat(progress);
    const empty = '░'.repeat(length - progress);
    const percentage = Math.round((current / total) * 100);
    
    return `${filled}${empty} ${percentage}%`;
}

export interface SlackFile {
    id: string;
    name: string;
    url_private: string;
    size?: number;
    filetype?: string;
    mimetype?: string;
}

export interface SlackMessage {
    files?: SlackFile[];
    text?: string;
    user?: string;
    ts?: string;
}
