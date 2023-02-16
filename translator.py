import torch


from model.transformer import Transformer



def translate_sentence(sentence, src_field, trg_field,
                       params,
                       max_len=50,
                       logging=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(params['src_vocab_size'],
                        params['trg_vocab_size'],
                        params['src_pad_idx'],
                        params['trg_pad_idx'],
                        params['embed_size'],
                        params['num_layers'],
                        params['forward_expansion'],
                        params['heads'],
                        params['dropout'],
                        params['max_length'],
                        params['device']).to(device)                
    model.load_state_dict(torch.load('transformer_ko_to_en.pt'))
    model.eval()

    if isinstance(sentence, str): # 문자열을 입력받으면
        tokenizer = src_tokenizer
        tokens = tokenizer.tokenize(sentence)
    else:
        tokens = sentence

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    if logging == True:
        print(f'전체 소스토큰 : {tokens}')


    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    if logging == True:
        print(f'소스 문장 인덱스: {src_indexes}')

    # into tensor 
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # mask src mask 
    src_mask = model.make_src_mask(src_tensor)


    with torch.no_grad():
        enc_out = model.encoder(src_tensor, src_mask)


    # 처음엔 시작토큰만 
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]


    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        # make target mask 
        trg_mask = model.make_trg_mask(trg_tensor)


        # encoder out 과 함께 decoder에서 attention 
        with torch.no_grad():
          output = model.decoder(trg_tensor, enc_out, trg_mask, src_mask)

        # last word token of output  
        pred_token = output.argmax(2)[:, -1].item()

        # append last token to index 
        trg_indexes.append(pred_token)

        # <EOS> token 만나면 종료  
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # index_to_sentence
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]
