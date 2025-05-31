import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(input_path, output_path):
    """
    Fungsi untuk memproses dataset secara otomatis.
    Input: Path ke file dataset mentah (CSV).
    Output: Dataset yang sudah diproses disimpan ke output_path.
    """
    # Memuat dataset
    df = pd.read_csv(input_path)
    
    # Feature engineering
    df['productivity_ratio'] = df['study_hours_per_day'] / (df['social_media_hours'] + df['netflix_hours'] + 1)
    bins = [0, 6, 8, float('inf')]
    labels = ['Kurang', 'Cukup', 'Berlebih']
    df['sleep_category'] = pd.cut(df['sleep_hours'], bins=bins, labels=labels, include_lowest=True)
    
    # Menangani missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df['parental_education_level'] = imputer.fit_transform(df[['parental_education_level']]).ravel()
    
    # Encoding kolom kategorikal
    categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                        'internet_quality', 'extracurricular_participation', 'sleep_category']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Menghapus kolom student_id
    df = df.drop('student_id', axis=1)
    
    # Normalisasi fitur numerik
    numeric_cols = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 
                    'attendance_percentage', 'sleep_hours', 'exercise_frequency', 
                    'mental_health_rating', 'productivity_ratio']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Menyimpan dataset
    df.to_csv(output_path, index=False)
    print(f"Dataset yang sudah diproses disimpan sebagai {output_path}")
    
    return df

if __name__ == "__main__":
    input_path = 'student_habits_performance_raw.csv'
    output_path = 'preprocessing/student_habits_preprocessing.csv'
    preprocess_data(input_path, output_path)
