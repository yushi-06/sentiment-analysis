# Sentiment Analysis Tool

A comprehensive sentiment analysis tool that can analyze text in two modes: manual input or dataset processing.

## Features

✅ **Manual Mode**: Enter text manually and get individual sentiment classifications  
✅ **Dataset Mode**: Process CSV/Excel files and get percentage distributions  
✅ **Three Classes**: Positive, Negative, Neutral  
✅ **High Accuracy**: Uses fine-tuned XLM-RoBERTa model  

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   venv\Scripts\activate
   ```

2. **Run the main tool:**
   ```bash
   python main.py
   ```

## Usage Modes

### Manual Mode
- Enter individual texts one by one
- Get immediate classification for each text
- Output: Only the label (Positive/Negative/Neutral)

**Example:**
```
Manual Mode:
Enter text: I love this product!
Enter text: This is terrible
Enter text: done

Results:
1. Positive
2. Negative
```

### Dataset Mode  
- Process entire CSV or Excel files
- Automatically finds "text" column or asks for column name
- Output: Overall percentage distribution

**Example:**
```
Dataset Mode:
Enter dataset file path: data.csv

Results:
Positive: 45.67%
Negative: 23.45%
Neutral: 30.88%
```

## File Formats Supported

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)

## Column Requirements

- Default column name: `text`
- If "text" column not found, you'll be prompted to specify the correct column name
- The tool will show available columns if needed

## Sample Files

- `sample_data.csv` - Example CSV file for testing dataset mode
- Contains 10 sample texts with various sentiments

## Other Scripts (Legacy)

- `infer_sentiment.py` - Shows detailed probabilities (for development)
- `train.py` - Model training script
- `test.py` - Model evaluation script
- `interactive_test.py` - Interactive testing with percentages

## Error Handling

- **"No data to analyze."** - No valid input provided
- **File not found** - Invalid file path
- **Column not found** - Specified column doesn't exist in dataset
- **Unsupported format** - File format not supported

## Requirements

The tool automatically loads all required dependencies from the virtual environment:
- torch
- transformers
- pandas
- openpyxl (for Excel support)

## Model Information

- **Base Model**: XLM-RoBERTa
- **Classes**: 3 (Positive, Negative, Neutral)  
- **Accuracy**: 92% on test set
- **Model Location**: `./sentiment_model/`

## Tips

1. **For single texts**: Use manual mode
2. **For bulk analysis**: Use dataset mode
3. **File paths**: You can drag and drop files to get the path
4. **Empty texts**: Automatically filtered out
5. **Percentages**: Rounded to 2 decimal places in dataset mode
