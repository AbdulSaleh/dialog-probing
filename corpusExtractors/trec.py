def main():
  corpus = open("corpora/rawCorpora/train_5500.label", 'r').readlines()
  processed_corpus_file = open(
      "corpora/processedCorpora/train_5500.label", 'w')
  output = ''
  for i, line in enumerate(corpus):
    question = line[line.index(' ') + 1:].rstrip()
    seperator = '' if i == len(corpus) - 1 else '\tepisode_done:True\t'
    output += f'text:{question}{seperator}'
  processed_corpus_file.write(output)


if __name__ == "__main__":
  main()
