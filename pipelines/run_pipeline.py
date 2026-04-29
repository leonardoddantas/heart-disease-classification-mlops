from src.data_loading import load_data

def run():
    df = load_data()
    print(df.head())

if __name__ == "__main__":
    run()