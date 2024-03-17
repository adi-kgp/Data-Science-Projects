import os 

instr_dict = {}

for instrument in os.listdir('/home/johnadi/Desktop/data-science-projects/audio-classification-deep-learning/wavfiles'):
    instr_dict[instrument] = os.listdir(f'/home/johnadi/Desktop/data-science-projects/audio-classification-deep-learning/wavfiles/{instrument}')

df = pd.DataFrame(instr_dict)
df2 = pd.melt(df)
df2.columns = ['label', 'fname']

df2.to_csv('instruments.csv', index=False)
