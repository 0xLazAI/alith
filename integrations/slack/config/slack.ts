/**
 * Configuration settings for the        messages: {
            title: 'LazAI Bot Commands',
            description: 'Easy data contribution to earn DAT tokens!'
        },
        errors: {
            fileNotFound: 'No files found in recent messages. Please upload a file first!',
            fileTooLarge: 'File is too large (max 100MB)',
            uploadFailed: 'Failed to contribute file. Please try again.',
            generic: 'Something went wrong. Please try again.',
        },
        success: {
            contribution: 'Successfully contributed {filename}!\nFile ID: {fileId}\nReward: {reward} DAT',
            processing: 'Contributing {filename}...',
        }ot
 */

export const config = {
    // Bot settings
    bot: {
        name: 'LazAI Bot',
        version: '1.0.0',
        maxFileSize: 100 * 1024 * 1024, // 100MB
        supportedFileTypes: ['.txt', '.json', '.csv', '.jsonl', '.parquet', '.md'],
        defaultReward: 100, // Default DAT reward
    },

    // Slack settings
    slack: {
        socketMode: true,
        port: parseInt(process.env.PORT || '3000'),
        maxFilesPerCommand: 10,
        historyLimit: 20,
    },

    // LazAI settings
    lazai: {
        encryptionSeed: "Sign to retrieve your encryption key",
        maxRetries: 3,
        timeoutMs: 30000, // 30 seconds
    },

    // Logging
    logging: {
        level: process.env.NODE_ENV === 'development' ? 'debug' : 'info',
        enableFileUploads: true,
        enableMetrics: true,
    },

    // Messages
    messages: {
        help: {
            title: '🤖 LazAI Bot Commands',
            description: 'Easy data contribution to earn DAT tokens!'
        },
        errors: {
            fileNotFound: '📂 No files found in recent messages. Please upload a file first!',
            fileTooLarge: '❌ File is too large (max 100MB)',
            uploadFailed: '❌ Failed to contribute file. Please try again.',
            generic: '❌ Something went wrong. Please try again.',
        },
        success: {
            contribution: '✅ Successfully contributed {filename}!\n🆔 File ID: {fileId}\n💰 Reward: {reward} DAT',
            processing: '🔄 Contributing {filename}...',
        }
    },

    // Validation rules
    validation: {
        filename: {
            maxLength: 255,
            allowedChars: /^[a-zA-Z0-9._-]+$/,
        },
        fileSize: {
            min: 1, // 1 byte
            max: 100 * 1024 * 1024, // 100MB
        }
    }
};

export default config;
