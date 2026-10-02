# 🎓 **Dataset Training Guide**

## 🚀 **Quick Start Training**

### **Option 1: Quick Training (Recommended)**
```bash
# Train from your existing bias correction data
python quick_train.py fix_bias_data.csv

# Train from any CSV file
python quick_train.py your_dataset.csv

# Train with custom column names
python quick_train.py reviews.csv review_text sentiment_label
```

### **Option 2: Interactive Training System**
```bash
python dataset_trainer.py
```

## 📊 **Supported Dataset Formats**

### **CSV Format (Recommended)**
```csv
text,sentiment
"I love this product",positive
"This is terrible",negative
"It's okay",neutral
```

### **Excel Format (.xlsx, .xls)**
Same structure as CSV but in Excel format.

### **Column Names**
The system auto-detects common column names:
- **Text columns**: text, review, comment, message, content, sentence
- **Sentiment columns**: sentiment, label, emotion, polarity, class, target

## 🎯 **Sentiment Label Formats**

The system automatically converts various label formats:

### **Positive Labels**
- `positive`, `pos`, `1`, `good`, `happy`, `like`

### **Negative Labels**  
- `negative`, `neg`, `0`, `bad`, `sad`, `hate`

### **Neutral Labels**
- `neutral`, `neu`, `2`, `ok`, `okay`, `mixed`

## 🔧 **Training Methods**

### **1. Quick Training Script**
```bash
python quick_train.py <file> [text_column] [sentiment_column]
```

**Features:**
- ✅ Fast batch processing
- ✅ Progress tracking
- ✅ Auto-saves model
- ✅ Works with enhanced or basic system

**Example:**
```bash
python quick_train.py movie_reviews.csv review rating
```

### **2. Interactive Training System**
```bash
python dataset_trainer.py
```

**Features:**
- ✅ Multiple file support
- ✅ Label mapping assistance
- ✅ Detailed statistics
- ✅ Batch processing options

### **3. Training from Fixed Main**
Use the enhanced `fixed_main.py` interactive learning mode:
```bash
python fixed_main.py
# Choose option 5: Interactive learning mode
```

## 📈 **Training Process**

### **What Happens During Training:**

1. **Load Dataset** - Reads CSV/Excel files
2. **Detect Columns** - Auto-finds text and sentiment columns
3. **Normalize Labels** - Converts labels to standard format
4. **Batch Processing** - Trains in batches for efficiency
5. **Adaptive Learning** - Uses confidence-based learning rates
6. **Save Progress** - Automatically saves improved model

### **Enhanced vs Basic Training:**

**Enhanced Training (Recommended):**
- Uses confidence-based adaptive learning rates
- Learns faster from uncertain predictions
- Provides detailed progress tracking
- Better handling of difficult examples

**Basic Training:**
- Standard adaptive learning
- Works without enhanced dependencies
- Simpler but effective

## 🎯 **Training Examples**

### **Example 1: Movie Reviews**
```csv
review,sentiment
"This movie is amazing!",positive
"Worst film ever",negative
"It was okay",neutral
```

```bash
python quick_train.py movie_reviews.csv review sentiment
```

### **Example 2: Product Reviews**
```csv
text,label
"Love this product",1
"Hate it",0
"Average quality",2
```

```bash
python quick_train.py products.csv text label
```

### **Example 3: Social Media Posts**
```csv
post,emotion
"Having a great day!",happy
"This is frustrating",angry
"Just normal day",neutral
```

```bash
python quick_train.py social.csv post emotion
```

## 📊 **Training Tips**

### **🎯 Best Practices:**

1. **Balanced Data**: Include roughly equal amounts of positive, negative, and neutral examples
2. **Quality over Quantity**: Clean, well-labeled data is better than large messy datasets
3. **Diverse Examples**: Include various writing styles, topics, and contexts
4. **Regular Training**: Retrain periodically with new data

### **⚡ Performance Tips:**

1. **Batch Size**: Default 100 examples per batch works well
2. **File Size**: Large files (>10MB) may take longer - consider splitting
3. **Memory**: Close other applications for large dataset training
4. **Progress**: Watch for progress updates every 100 examples

### **🔍 Quality Checks:**

1. **Test After Training**: Use `python fixed_main.py` to test predictions
2. **Check Statistics**: Review success/failure rates
3. **Validate Results**: Test with known examples
4. **Monitor Confidence**: Watch for improved confidence scores

## 🚀 **Complete Training Workflow**

### **Step 1: Prepare Your Dataset**
```csv
text,sentiment
"Your training examples here",positive
```

### **Step 2: Quick Train**
```bash
python quick_train.py your_data.csv
```

### **Step 3: Test Results**
```bash
python fixed_main.py
# Choose option 1 or 2 to test predictions
```

### **Step 4: Validate Performance**
```bash
python fixed_main.py
# Choose option 6 to see statistics
```

## 🎉 **Training Success Indicators**

### **✅ Good Training Results:**
- Success rate > 90%
- Balanced learning across all sentiment types
- Improved confidence scores on test examples
- Better predictions on your specific domain

### **⚠️ Issues to Watch:**
- High failure rate (>10%)
- Skewed towards one sentiment type
- Low confidence scores after training
- Poor performance on test cases

## 🔧 **Troubleshooting**

### **Common Issues:**

**"Column not found"**
- Check your column names
- Use quotes if column names have spaces
- Try the interactive trainer for assistance

**"No valid examples"**
- Check for empty rows or invalid labels
- Ensure text column has actual text content
- Verify sentiment labels are recognized

**"Training failed"**
- Check file format (CSV/Excel)
- Ensure file is not corrupted
- Try with a smaller sample first

### **Getting Help:**
1. Use interactive trainer: `python dataset_trainer.py`
2. Check training statistics for insights
3. Test with small sample datasets first
4. Verify your data format matches examples

---

## 🎯 **Ready to Train!**

**Quick Start:**
```bash
python quick_train.py fix_bias_data.csv
```

**Your model will be automatically updated and ready to use with improved accuracy!** 🚀
