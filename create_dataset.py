from src.preprocessing.processor import Processor


if __name__ == "__main__":
    processor = Processor({}, conf="settings.toml")
    processor.process_dataset()
