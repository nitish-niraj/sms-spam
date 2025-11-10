"""Quick EDA before training"""
import pandas as pd

# Load dataset
df = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['label', 'message']) # pyright: ignore[reportUnknownMemberType]

print('='*60)
print('DATASET QUICK CHECK')
print('='*60)
print(f'\nTotal messages: {len(df)}')
print(f'Spam: {(df.label=="spam").sum()} ({(df.label=="spam").sum()/len(df)*100:.1f}%)')
print(f'Ham: {(df.label=="ham").sum()} ({(df.label=="ham").sum()/len(df)*100:.1f}%)')
print(f'\nAvg message length: {df.message.str.len().mean():.1f} chars')
print(f'Max message length: {df.message.str.len().max()} chars')
print(f'Min message length: {df.message.str.len().min()} chars')

print('\n' + '='*60)
print('SAMPLE MESSAGES')
print('='*60)
print('\nSpam samples:')
for i, msg in enumerate(df[df.label=='spam'].message.head(3), 1):
    print(f'{i}. {msg[:80]}...' if len(msg) > 80 else f'{i}. {msg}')

print('\nHam samples:')
for i, msg in enumerate(df[df.label=='ham'].message.head(3), 1):
    print(f'{i}. {msg[:80]}...' if len(msg) > 80 else f'{i}. {msg}')

print('\n' + '='*60)
print('READY TO TRAIN!')
print('='*60)
