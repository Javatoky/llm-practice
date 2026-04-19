import asyncio
import json

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL


class NewArticle(BaseModel):
    title: str = Field(description = "标题")
    source: str = Field(description = "来源")
    release_date: str = Field(description = "提交日期")
    key_role: list[str] = Field(description = "关键人物")
    event_summary: str = Field(description = "事件摘要")

class ProductDescription(BaseModel):
    product_name: str = Field(description = "产品名")
    brand: str = Field(description = "品牌")
    price: float = Field(ge = 0, description = "价格")
    key_selling_point: list[str] = Field(description = "核心卖点列表")
    target_audience: list[str] = Field(description = "适用人群")

model_dict = {
    NewArticle.__name__: NewArticle,
    ProductDescription.__name__: ProductDescription
}

async def extract_info(text: str, model_class: type[BaseModel]) -> BaseModel:
    schema = model_class.model_json_schema()
    root_prompt = f"""
    # Role: 你是一个**信息提取器**，能根据要求的文本类型在用户输入的文本中提取所要求的字段

    # Capabilities & Constrains: 
    - 你只能对一个**{model_class.__name__}类型文本**进行字段提取工作
    - 任何超出职责的工作（任何非相关的其他提问：**图片要点**提取、**文本总结**工作、**新闻搜索**工作、**产品评价**工作等等）必须用一句话干脆的拒绝
    - 不得编造输入文本的中没有的信息，必须依据文本的实际内容提取

    # Instructions:
    - 请从用户提供的文本中提取信息，以 JSON 格式返回
    - 请严格按照 Knowledge 中的 JSON Schema 定义返回JSON
    - 如果某个字段在文本中无法提取，将其设为'未知'
    # Knowledge: 
        - 用户要求的 JSON Schema 定义：
            {schema}

    # Output Format:
    - 只按Instructions中的示例回复 **JSON格式** 的提取信息
    """.strip()

    client = AsyncOpenAI(
        api_key = QWEN_API_KEY,
        base_url = QWEN_BASE_URL
    )

    messages = [
        {"role": "system", "content": root_prompt},
        {"role": "user", "content": text}
    ]

    async def chat(messages: list[dict]) -> str:
        response = await client.chat.completions.create(
            model = QWEN_MODEL,
            messages = messages,
            response_format = {"type": "json_object"}
        )
        reply = response.choices[0].message.content
        return reply



    async def field_validation() -> BaseModel:
        reply = await chat(messages)
        retries_messages = list(messages)
        for retries_count in range(0, 4):
            try:
                parse_reply = json.loads(reply)
                result = model_class.model_validate(parse_reply)
            except json.JSONDecodeError as e:
                print(f"JSON 语法错误：{e}")
                break
            except ValidationError as e:
                if retries_count == 3:
                    raise
                error_msg = f"格式错误：{e}，请重新生成!"
                retries_messages.append({"role": "assistant", "content": reply})
                retries_messages.append({"role": "user", "content": error_msg})
                reply = await chat(retries_messages)
            else:
                messages.append({"role": "assistant", "content": reply})
                print(f'重试次数：{retries_count}')
                return result
    return await field_validation()

test_cases = [
    {
        "type": "new_article",
        "text": "近日，科技巨头苹果公司发布了最新款iPhone 15，由CEO蒂姆·库克在加州总部正式宣布。此次发布会重点介绍了新机搭载的A17芯片和革命性的相机系统，吸引了全球媒体和消费者的广泛关注。据悉，新产品将于下月正式上市销售。",
    },
    {
        "type": "new_article", 
        "text": "昨日晚间，著名物理学家杨振宁教授在北京大学发表演讲，深入探讨了现代物理学的发展趋势。多位知名学者参与了讨论，包括中科院院士李政道教授。此次学术交流活动为推动我国物理学研究发展起到了积极作用。",
    },
    {
        "type": "product_description",
        "text": "全新小米智能手环8 Pro，小米品牌出品。采用AMOLED高清彩屏，支持全天心率监测、血氧饱和度检测，拥有超过100种运动模式。防水等级达到5ATM，续航长达14天。售价仅需299元，是健康管理的绝佳选择，特别适合运动爱好者和关注健康的人群。",
    },
    {
        "type": "product_description",
        "text": "戴尔XPS 13笔记本电脑，戴尔官方出品。搭载第13代英特尔酷睿处理器，16GB内存，512GB固态硬盘，13.4英寸触控屏幕。轻薄便携设计，重量仅为1.2kg，电池续航长达12小时。售价8999元，性能强劲，适合商务人士和创意工作者使用。",
    }
]

async def main():
    for i, case in enumerate(test_cases, 1):
        print(f'---Test{i}---')
        try:
            result = await extract_info(case["text"], model_dict[case["type"].strip()])
        except ValidationError as e:
            print(f"提取失败，格式错误：{e}")
        else:
            print(f'提取成功：{result}')
            print("\n")
    print(f"---测试完毕---")

if __name__ == "__main__":
    asyncio.run(main())


# 错误总结：
# 1. 类名命名不规范：未采用大驼峰命名（知识点缺失）
# 2. import放在函数内容，应该同一放在文件顶部，对Python规范不了解（知识点缺失）
# 3. 函数嵌套过深，整个extract——info的逻辑全在field——validation，没必要嵌套，chat可以独立出去，extract——info只负责调用就可以，原则：扁平结构比嵌套结构更易读（设计思想问题）
# 4. 测试用例都太过完整，应该设几个信息缺失和职责边界外等等例子（LLM测试思想：质量低的测试用例可以评判自己的设计是否周全）