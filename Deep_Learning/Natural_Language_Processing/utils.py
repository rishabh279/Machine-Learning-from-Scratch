import torch
import spacy



def translate_sentence(model, sentence, german, english, device, max_length=50, attention=False):
    spacy_ger = spacy.load('de')

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    if not attention:
        with torch.no_grad():
            hidden, cell = model.encoder(sentence_tensor)
    else:
        with torch.no_grad():
            encoder_states, hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi['<sos>']]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)
        if not attention:
            with torch.no_grad():
                output, hidden, cell = model.decoder(previous_word,
                                                     hidden, cell)
        else:
            with torch.no_grad():
                output, hidden, cell = model.decoder(previous_word, encoder_states,
                                                     hidden, cell)
        best_guess = output.argmax(1).item()
        outputs.append(best_guess)

        if output.argmax(1).item() == english.vocab.stoi['<eos>']:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])