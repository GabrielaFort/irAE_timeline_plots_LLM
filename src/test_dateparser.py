from dateparser import parse
import dateparser

print("Testing dateparser with various date formats:")
print(parse('December 2015'))

print(parse('Dec 2015'))

print(parse('2015'))

print(parse('early 2015'))

print(parse('late 2015'))

print(parse('2015 early'))

print(parse('1991-05-17'))

print(parse('17 May 1991'))

print(parse('1991/05/17'))

print(parse('05/17/1991'))

print(parse('05-17-1991'))

print((parse('04/17/25')).strftime("%m/%d/%Y"))

print((parse("2025-07")).strftime("%m/%d/%Y"))

print((parse("17-APR-2018")).strftime("%m/%d/%Y"))

print((parse("2019-spring")).strftime("%m/%d/%Y"))
