from datasets import load_dataset

def main():
    print("Hello from decoupling!")
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    print(dataset)


if __name__ == "__main__":
    main()
