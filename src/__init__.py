# class Solution:
#     def dailyTemperatures(self, temperatures):
#         result = []
#         for index, temp in enumerate(temperatures):
#             print(f"temp = {temp}")
#             for temp1 in temperatures[index:]:
#                 if temp1 > temp:
#                     print(temp1)
#                     result.append(temperatures.index(temp1))
#                     break
#         print(result)
# s = Solution()
# s.dailyTemperatures([30,38,30,36,35,40,28])
        