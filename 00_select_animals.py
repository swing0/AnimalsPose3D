import os
import shutil
from pathlib import Path

# 目标四足哺乳动物列表 (允许的科属)
# 只保留标准的四足行走动物：猫科, 犬科, 熊科, 牛科, 鹿科, 羊科, 骆驼科, 猪科, 马科, 犀科, 象科, 鬣狗科, 河马科
ALLOWED_FAMILIES = {
    'Felidae': '猫科',
    'Canidae': '犬科',
    'Ursidae': '熊科',
    'Bovidae': '牛科',
    'Cervidae': '鹿科',
    'Hippopotamidae': '河马科',
    'Camelidae': '骆驼科',
    'Suidae': '猪科',
    'Equidae': '马科',
    'Rhinocerotidae': '犀科',
    'Elephantidae': '象科',
    'Hyaenidae': '鬣狗科',
    'Mustelidae': '鼬科 (部分大型)', # 如狼獾
}

# 详细的翻译字典 (仅包含允许的四足动物)
mammal_translations = {
    # --- 猫科 Felidae ---
    'African_Leopard': '非洲豹',
    'Amur_Leopard': '东北豹',
    'Bengal_Tiger': '孟加拉虎',
    'Caracal': '狞猫', # 中型
    'Cheetah': '猎豹',
    'Clouded_Leopard': '云豹',
    'Cougar': '美洲狮',
    'Eurasian_Lynx': '欧亚猞猁',
    'Jaguar': '美洲虎',
    'Ocelot': '虎猫',
    'Pallas_Cat': '兔狲',
    'Sand_Cat': '沙猫',
    'Siberian_Tiger': '东北虎',
    'Snow_Leopard': '雪豹',
    'West_African_Lion': '西非狮',
    
    # --- 犬科 Canidae ---
    'African_Wild_Dog': '非洲野犬',
    'Arctic_Fox': '北极狐',
    'Arctic_Wolf': '北极狼',
    'Bush_Dog': '薮犬',
    'Coyote': '郊狼',
    'Dhole': '豺',
    'Dingo': '澳洲野犬',
    'Fennec_Fox': '耳廓狐',
    'Gray_Wolf': '灰狼',
    'Japanese_Raccoon_Dog': '日本貉',
    'Maned_Wolf': '鬃狼',
    'Red_Fox': '赤狐',

    # --- 熊科 Ursidae ---
    'Formosan_Black_Bear': '台湾黑熊',
    'Giant_Panda': '大熊猫',
    'Grizzly_Bear': '灰熊',
    'Himalayan_Brown_Bear': '喜马拉雅棕熊',
    'Polar_Bear': '北极熊',
    'Sloth_Bear': '懒熊',
    'Spectacled_Bear': '眼镜熊',
    'Sun_Bear': '马来熊',

    # --- 牛科 Bovidae ---
    'Addax': '旋角羚',
    'African_Buffalo': '非洲水牛',
    'Alpine_Goat': '阿尔卑斯山羊',
    'Alpine_Ibex': '阿尔卑斯羱羊',
    'American_Bison': '美洲野牛',
    'Bighorn_Sheep': '大角羊',
    'Black_Wildebeest': '黑角马',
    'Blackbuck': '印度黑羚',
    'Blue_Wildebeest': '蓝角马',
    'Bongo': '紫羚',
    'Dall_Sheep': '白大角羊',
    'Dama_Gazelle': '鹿羚',
    'Gemsbok': '南非剑羚',
    'Highland_Cattle': '高地牛',
    'Hill_Radnor_Sheep': '希尔拉德诺羊',
    'Kirks_Dik_Dik': '柯氏犬羚',
    'Markhor': '捻角山羊',
    'Nile_Lechwe': '尼罗苇羚',
    'Nilgai': '蓝牛羚',
    'Nyala': '林羚',
    'Pronghorn_Antelope': '叉角羚',
    'Sable_Antelope': '貂羚',
    'Saiga': '赛加羚',
    'Scimitar_Horned_Oryx': '弯角剑羚',
    'Springbok': '跳羚',
    'Takin': '羚牛',
    'Thomsons_Gazelle': '汤姆森瞪羚',
    'Wild_Water_Buffalo': '野生水牛',
    'Wisent': '欧洲野牛',

    # --- 鹿科 Cervidae ---
    'Alaskan_Moose': '阿拉斯加驼鹿',
    'Bairds_Tapir': '贝氏貘', 
    'European_Fallow_Deer': '欧洲黇鹿',
    'Malayan_Tapir': '马来貘',
    'Okapi': '㺢㹢狓', 
    'Pere_Davids_Deer': '麋鹿',
    'Red_Deer': '马鹿',
    'Reindeer': '驯鹿',
    'Reticulated_Giraffe': '网纹长颈鹿', 

    # --- 其他有蹄类 (马科 Equidae, 猪科 Suidae, 河马科, setc.) ---
    'Babirusa': '鹿豚',
    'Collared_Peccary': '领西猯',
    'Common_Warthog': '疣猪',
    # 'Grevy_Zebra': '细纹斑马', # Not in original list but common
    'Hippopotamus': '河马',
    'Pygmy_Hippo': '倭河马',
    'Plains_Zebra': '平原斑马',
    'Przewalskis_Horse': '普氏野马',
    'Red_River_Hog': '红河猪',
    'Somali_Wild_Ass': '索马里野驴',
    'Standard_Donkey': '标准驴',
    'Tamworth_Pig': '塔姆沃思猪',
    'Wild_Boar': '野猪',

    # --- 骆驼科 Camelidae ---
    'Alpaca': '羊驼',
    'Bactrian_Camel': '双峰驼',
    'Dromedary_Camel': '单峰驼',
    'Llama': '美洲驼',
    
    # --- 象科 ---
    'African_Elephant': '非洲象',
    # 'Asian_Elephant': '亚洲象', # Original list logic check
    'Bornean_Pygmy_Elephant': '婆罗洲侏儒象',
    'Indian_Elephant': '印度象',

    # --- 犀科 ---
    'Black_Rhino': '黑犀牛',
    'Indian_Rhinoceros': '印度犀',
    'Southern_White_Rhinoceros': '南方白犀',

    # --- 鬣狗科 Hyaenidae ---
    'Spotted_Hyena': '斑鬣狗',
    'Striped_Hyena': '条纹鬣狗',

    # --- 鼬科 Mustelidae (大型) ---
    'Wolverine': '狼獾', 
    'European_Badger': '欧洲獾',
    'Honey_Badger': '蜜獾',
    
    # --- 灵猫科/獴科/其他食肉目 ---
    'Fossa': '马岛獴', 
    'Binturong': '熊狸',
}

# 需要排除的特定动物 (即便在上面列表中可能是误选，或者是非四足行走的哺乳动物)
EXCLUDE_ANIMALS = {
    # 灵长类 (Primates) - 绝对排除
    'B_W_Ruffed_Lemur', 'Bonobo', 'Bornean_Orangutan', 'Capuchin_Monkey', 
    'Chimpanzee', 'Coquerels_Sifaka', 'Gorilla', 'Hamadryas_Baboon', 
    'Human', 'Japanese_Macaque', 'Lar_Gibbon', 'Lemur', 'Lion_Tailed_Macaque',
    'Mandrill', 'Proboscis_Monkey', 'Red_Ruffed_Lemur', 'Ring_Tailed_Lemur',
    'Siamang', 'Western_Chimpanzee', 'Western_Lowland_Gorilla', 'White_Faced_Saki',
    
    # 有袋类 (Marsupials) - 跳跃行走，排除
    'Common_Wombat', 'Koala', 'Quokka', 'Red_Kangaroo', 'Rednecked_Wallaby', 
    'Tasmanian_Devil',
    
    # 水生/半水生且姿态差异大的
    'California_Sea_Lion', 'Grey_Seal', 'Platypus', 
    'Asian_Small_Clawed_Otter', 'Giant_Otter', 
    
    # 其他非标准四足
    'Nine_Banded_Armadillo', 'Chinese_Pangolin', 'Giant_Anteater', 'Meerkat', 'Prairie_Dog',
    'Raccoon', 'Striped_Skunk', 'North_American_Beaver', 'Capybara', 'African_Crested_Porcupine'
}

def get_current_time():
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_mammal_translation_file(moved_files, target_dir):
    """生成哺乳动物中英文对照文件"""
    # 按动物名分组，去除重复
    animals_dict = {}
    for item in moved_files:
        animals_dict[item['name']] = mammal_translations[item['name']]

    # 按英文名排序
    sorted_animals = sorted(animals_dict.items(), key=lambda x: x[0])

    # 生成对照文件内容
    content = []
    content.append("哺乳动物中英文对照表 - 标准四足动物")
    content.append("=" * 70)
    content.append("生成时间：" + get_current_time())
    content.append("=" * 70)
    content.append("")
    content.append(f"{'英文名':<25} {'中文名':<20}")
    content.append("-" * 80)

    for animal_en, animal_cn in sorted_animals:
        content.append(f"{animal_en:<25} {animal_cn:<20}")

    # 写入文件
    output_file = target_dir / "mammals_translation.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        print(f"\n✓ 已生成哺乳动物中英文对照文件: {output_file}")
    except Exception as e:
        print(f"Warning: Could not write translation file: {e}")

def move_mammal_files():
    """移动筛选后的标准四足哺乳动物文件"""

    # 目录路径 - 确认为用户指定的路径
    source_dir = Path(r"D:\workplace\export_json_loop")
    target_dir = Path(r"D:\workplace\export_json_test")

    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")

    # 确保目标目录存在
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create target directory (permission error?): {e}")

    # 统计信息
    moved_files = []
    skipped_files = []
    
    # 获取文件列表
    if os.path.exists(source_dir):
        file_list = os.listdir(source_dir)
    else:
        print(f"\nError: Source directory {source_dir} not found. Please check path.")
        return

    print("=" * 60)
    print("开始筛选和移动...")
    print("条件：标准四足哺乳动物 (Cats, Dogs, Bears, Horses, Cows, etc.)")
    print("=" * 60)

    for filename in file_list:
        if filename.endswith('.ovl.zip'):
            base_name = filename[:-8]
            
            # 分离动物名
            animal_name = None
            for suffix in ['_Female', '_Juvenile', '_Male']:
                if base_name.endswith(suffix):
                    animal_name = base_name[:-len(suffix)]
                    break
            if animal_name is None:
                animal_name = base_name

            # 1. 检查是否在排除列表中
            if animal_name in EXCLUDE_ANIMALS:
                skipped_files.append({'name': animal_name, 'reason': 'Excluded (Primate/Marsupial/Other)'})
                continue
            
            # 2. 检查是否在允许的翻译字典中
            if animal_name not in mammal_translations:
                skipped_files.append({'name': animal_name, 'reason': 'Not in Allowlist'})
                continue
                
            # 执行移动
            source_file = source_dir / filename
            target_file = target_dir / filename
            
            try:
                shutil.move(str(source_file), str(target_file))
                moved_files.append({'name': animal_name})
                print(f"✓ [KEEP] {animal_name} ({mammal_translations[animal_name]})")
            except Exception as e:
                print(f"Error moving {filename}: {e}")

    # 生成报告
    print("\n" + "=" * 60)
    print(f"操作完成!")
    print(f"成功移动: {len(moved_files)} 个文件")
    print(f"跳过/排除: {len(skipped_files)} 个文件")
    
    if moved_files:
        generate_mammal_translation_file(moved_files, target_dir)

def main():
    print("通用四足动物筛选程序 (Standard Quadruped Selection)")
    move_mammal_files()

if __name__ == "__main__":
    main()