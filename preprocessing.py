

# Convert all entries to lowercase
df.loc[:, df.columns != 'post_time'] = df.loc[:, df.columns != 'post_time'].applymap(lambda x: x.lower() if
                                                                           isinstance(x, str) else x)
df.head(1)
