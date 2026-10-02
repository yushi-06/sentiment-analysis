# 📂 **Dataset Addresses & Training Guide**

## 🎯 **Your Dataset Training System**

You now have a complete dataset management and training system with pre-configured dataset addresses!

## 📊 **Pre-Configured Datasets**

### **1. Your Original Bias Correction Data**
- **Name**: `bias_correction`
- **File**: `fix_bias_data.csv`
- **Columns**: `text` → `sentiment`
- **Description**: Your original 18,734+ bias correction examples

### **2. Sample Movie Reviews**
- **Name**: `movie_reviews`
- **File**: `sample_movie_reviews.csv`
- **Columns**: `review_text` → `sentiment_label`
- **Description**: 20 sample movie reviews (positive/negative/neutral)

### **3. Sample Product Reviews**
- **Name**: `product_reviews`
- **File**: `sample_product_reviews.csv`
- **Columns**: `text` → `sentiment`
- **Description**: 20 sample product reviews

### **4. Sample Social Media Posts**
- **Name**: `social_media`
- **File**: `sample_social_media.csv`
- **Columns**: `post` → `emotion`
- **Description**: 20 sample social media posts

## 🚀 **Easy Training Options**

### **Option 1: Training Menu (Easiest)**
```bash
python train_menu.py
```
- Shows all available datasets
- One-click training from any dataset
- Add new datasets interactively

### **Option 2: Quick Training Commands**
```bash
# Train from your bias correction data
python quick_train.py fix_bias_data.csv text sentiment

# Train from movie reviews
python quick_train.py sample_movie_reviews.csv review_text sentiment_label

# Train from product reviews
python quick_train.py sample_product_reviews.csv text sentiment

# Train from social media posts
python quick_train.py sample_social_media.csv post emotion
```

### **Option 3: Dataset Configuration Manager**
```bash
python dataset_config.py
```
- Add/remove datasets
- Manage dataset configurations
- Update dataset status

## 📝 **Adding Your Own Datasets**

### **Method 1: Through Training Menu**
```bash
python train_menu.py
# Choose "Add new dataset" option
```

### **Method 2: Through Configuration Manager**
```bash
python dataset_config.py
# Choose option 1: "Add new dataset"
```

### **Method 3: Direct Training**
```bash
python quick_train.py your_file.csv text_column sentiment_column
```

## 🎯 **Dataset Format Requirements**

Your datasets should be CSV or Excel files with:

### **Required Columns:**
- **Text column**: Contains the text to analyze
- **Sentiment column**: Contains the sentiment labels

### **Example CSV Format:**
```csv
text,sentiment
"I love this product",positive
"This is terrible",negative
"It's okay",neutral
```

### **Supported Label Formats:**
- **Positive**: positive, pos, 1, good, happy, like
- **Negative**: negative, neg, 0, bad, sad, hate
- **Neutral**: neutral, neu, 2, ok, okay, mixed

## 🔧 **System Files Created**

### **Configuration Files:**
- `dataset_paths.json` - Stores all dataset configurations
- `dataset_config.py` - Dataset configuration manager
- `train_menu.py` - Easy training menu
- `setup_datasets.py` - Auto-setup sample datasets

### **Sample Datasets:**
- `sample_movie_reviews.csv` - Movie review examples
- `sample_product_reviews.csv` - Product review examples
- `sample_social_media.csv` - Social media post examples

### **Training Scripts:**
- `quick_train.py` - Fast training from any dataset
- `dataset_trainer.py` - Advanced training system

## 🎉 **Complete Workflow**

### **Step 1: View Available Datasets**
```bash
python train_menu.py
```

### **Step 2: Select Dataset to Train From**
Choose from the menu or add your own dataset

### **Step 3: Start Training**
The system will automatically:
- Load your dataset
- Detect columns
- Normalize labels
- Train the model
- Save improvements

### **Step 4: Test Results**
```bash
python fixed_main.py
# Test your improved model
```

## 🎯 **Quick Start Examples**

### **Train from Your Bias Data:**
```bash
python train_menu.py
# Choose option 1: bias_correction
```

### **Train from Movie Reviews:**
```bash
python train_menu.py
# Choose option 2: movie_reviews
```

### **Add Your Own Dataset:**
```bash
python train_menu.py
# Choose "Add new dataset"
# Enter your file path and column names
```

## 📊 **Benefits of This System**

✅ **Pre-configured datasets** - Ready to use sample data
✅ **Easy dataset management** - Add/remove datasets easily
✅ **Automatic column detection** - Smart column finding
✅ **Label normalization** - Converts various label formats
✅ **Progress tracking** - See training progress
✅ **Error handling** - Robust training process
✅ **Model auto-save** - Automatically saves improvements

## 🚀 **Ready to Use!**

Your dataset training system is now complete with pre-configured addresses and easy-to-use interfaces!

**Start training now:**
```bash
python train_menu.py
```

All your datasets are pre-configured and ready for training! 🎯
