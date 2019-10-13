 # .item(), .keys(), .values()
-------------------------------------------------------------------------------------------------------
.item(), .keys(), .values()都是python字典函数，三个函数都是查看字典中元素的函数，返回值都为一个list(列表）
.item()函数以列表返回可遍历的(键, 值) 元组数组,可以用于 for 来循环遍历
用法：dict.items()

e.g.
dict = {1:2,'a':'b','hello':'world'}
print(dict.items())

<<输出结果
dict_items([(1, 2), ('a', 'b'), ('hello', 'world')])

.keys()
e.g.
dict = {1:2,'a':'b','hello':'world'}
print(dict.keys())

<<
dict_keys([1, 'a', 'hello'])


.values()
e.g.
dict = {1:2,'a':'b','hello':'world'}
print(dict.values())

<<
dict_values([2, 'b', 'world'])
------------------------------------------------------------------------------------------------------------