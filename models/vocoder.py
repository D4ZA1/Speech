from torchaudio.models import wavernn

class Vocoder(nn.Module):
    def __init__(self):
        super(Vocoder, self).__init__()
        self.model = wavernn(pretrained=True)

    def forward(self, mel_spec):
        return self.model.infer(mel_spec)
