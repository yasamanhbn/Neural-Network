import train
import test

if __name__ == "__main__":
    model = train.trainProcess()
    print("test process is started")
    test.testProcess(model)