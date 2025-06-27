from src.preprocessing.processor import Processor


if __name__ == "__main__":
    processor = Processor({}, conf="settings.yaml")
    processor.process_dataset()
