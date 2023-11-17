import torch

# 목업 데이터 요소
num_songs = 500  # 노래 개수
codes_per_song = 100  # 노래당 코드 개수
numbers_per_code = 4  # 코드 1개당 표현

#  목업 데이터 생성
music_codes = torch.rand((num_songs, numbers_per_code, codes_per_song))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
music_codes = music_codes.to(device)


class Chord_CNN(torch.nn.Module):
    def __init__(self):
        super(Chord_CNN, self).__init__()  # 부모 클래스 초기화

        # 첫번째 합성곱 계층
        self.conv1 = torch.nn.Sequential(
            # output shape : [500, 64, 97]
            # 높이 4의 window -> 100 크기에 1씩 올려가며 적용 -> 97짜리 배열 -> 이짓을 64번
            torch.nn.Conv1d(in_channels=4, out_channels=64, kernel_size=4, stride=1),
            # ReLU 함수
            torch.nn.ReLU(),
            # 위의 97짜리 배열에서 윈도 3으로 max poling ->  97이 32됨
            torch.nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.flatten = torch.nn.Flatten()

        self.affine = torch.nn.Linear(64 * 32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.affine(x)
        return x


model = Chord_CNN().to(device)

output = model(music_codes)

print(output.shape)
