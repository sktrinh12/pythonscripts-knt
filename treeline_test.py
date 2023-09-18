class InputData:
    def __init__(self):
        # self.data = []
        self.sum = 0
        self.count = 0

    def calc_mean(self):
        assert self.count != 0, "no data"
        return self.sum / self.count

    def add_data(self, x):
        # self.data.append(x)
        self.sum += x
        self.count += 1


test_data = InputData()
for i in range(0, 4):  # O(n)
    test_data.add_data(i)  # O(1)

    assert test_data.calc_mean() == 2, f"error, {test_data.calc_mean()}"  # O(2*1)
