# HRM Training Data Expansion
# Generated for expanding character language training dataset

## Safety Notice
⚠️ **Full System Scan Opt-in Required**

By default, HRM only scans user-specified data directories. To enable comprehensive system-wide scanning (including all drives and user directories), set the environment variable:

```bash
export HRM_SCAN_FULL=1
```

This opt-in is required for privacy and security. Full scanning may take significant time and resources.

## Current Status
- Training sequences: 4 (minimum 256 chars each)
- Validation sequences: 4
- Total: 8 sequences (very limited)

## Target Goals
- Expand to 100+ training sequences
- Add diverse text sources
- Improve sequence variety
- Maintain character-level format

## Data Sources to Add

### 1. Literary Content
- Classic books (Project Gutenberg)
- Modern fiction
- Technical documentation
- Scientific papers

### 2. Code Content
- Python repositories
- JavaScript libraries
- C++ examples
- API documentation

### 3. Conversational Data
- Chat logs (anonymized)
- Forum discussions
- Q&A pairs
- Technical support conversations

### 4. Web Content
- Wikipedia articles
- Blog posts
- News articles
- Documentation

## Implementation Strategy

### Phase 1: Raw Data Collection
1. Create `data/text/raw/` directory structure
2. Add diverse text files
3. Ensure minimum 256 characters per sequence
4. Mix different writing styles and topics

### Phase 2: Data Processing
1. Update `prepare_language_dataset.sh` to handle new sources
2. Generate intelligent contexts from expanded data
3. Create train/validation/test splits
4. Verify character encoding works with new content

### Phase 3: Quality Control
1. Filter out low-quality sequences
2. Ensure vocabulary coverage
3. Balance different content types
4. Validate sequence lengths

## Next Actions
1. Create raw data directory structure
2. Add sample diverse content
3. Update preprocessing script
4. Test with expanded dataset

## Expected Results
- 100+ training sequences
- 20+ validation sequences  
- Better language model coverage
- Improved generalization capability