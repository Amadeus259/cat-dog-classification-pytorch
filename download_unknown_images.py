"""
download_unknown_images.py —— 下载“未知类别”测试图片
======================================================
第一阶段未知检测需要一个“模型没见过的类别”图片集，例如兔子。
本脚本用 icrawler 从百度图片/Bing 图片抓取指定数量的图片，
下载完成后自动做三件事：

1. 过滤打不开、损坏的图片；
2. 过滤尺寸过小的图片（默认短边至少 100 像素）；
3. 按文件内容去重，并统一重命名成 xxx_0001.jpg 这样的文件名。

用法示例：
    python download_unknown_images.py --out_dir data/unknown_set/rabbit --max_num 150
"""
import argparse
import hashlib
from pathlib import Path

from PIL import Image


def crawl_baidu(keyword: str, max_num: int, out_dir: Path, file_idx_offset: int = 0) -> None:
    """使用百度图片搜索抓取图片。"""
    from icrawler.builtin import BaiduImageCrawler

    crawler = BaiduImageCrawler(
        storage={'root_dir': str(out_dir)},
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=8,
    )
    crawler.crawl(keyword=keyword, max_num=max_num, file_idx_offset=file_idx_offset)


def crawl_bing(keyword: str, max_num: int, out_dir: Path, file_idx_offset: int = 0) -> None:
    """使用 Bing 图片搜索抓取图片（作为百度抓取不足时的补充）。"""
    from icrawler.builtin import BingImageCrawler

    crawler = BingImageCrawler(
        storage={'root_dir': str(out_dir)},
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=8,
    )
    crawler.crawl(keyword=keyword, max_num=max_num, file_idx_offset=file_idx_offset)


def count_files(directory: Path) -> int:
    return len([p for p in directory.iterdir() if p.is_file()])


def clean_images(directory: Path, prefix: str, min_side: int = 100) -> int:
    """
    清洗抓取结果：
    - 过滤损坏、无法解码的图片；
    - 过滤短边小于 min_side 的图片；
    - 按内容去重；
    - 统一转成 RGB 的 JPG 并重命名。
    返回清洗后保留的图片数量。
    """
    files = sorted(p for p in directory.iterdir() if p.is_file())
    valid_images = []
    seen_hashes = set()

    for path in files:
        try:
            # 用 hashlib 对原始字节去重，避免同一张图被重复下载
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                continue

            with Image.open(path) as im:
                im = im.convert('RGB')
                width, height = im.size
                if min(width, height) < min_side:
                    continue
                # 复制一份，后面统一保存
                valid_images.append(im.copy())
            seen_hashes.add(digest)
        except Exception:
            # 打不开或解码失败的图片直接跳过
            continue

    # 删除原始文件，避免旧文件和新命名文件混在一起
    for path in files:
        try:
            path.unlink()
        except Exception:
            pass

    for index, image in enumerate(valid_images, start=1):
        output_path = directory / f'{prefix}_{index:04d}.jpg'
        image.save(output_path, 'JPEG', quality=95)

    return len(valid_images)


def main():
    parser = argparse.ArgumentParser(description='下载未知类别图片（默认兔子）')
    parser.add_argument('--out_dir', type=str, default='data/unknown_set/rabbit',
                        help='图片保存目录')
    parser.add_argument('--max_num', type=int, default=150,
                        help='目标图片数量')
    parser.add_argument('--keyword_zh', type=str, default='兔子 真实照片',
                        help='百度图片搜索关键词')
    parser.add_argument('--keyword_en', type=str, default='rabbit animal photo',
                        help='Bing 图片搜索关键词（补充抓取时使用）')
    parser.add_argument('--min_side', type=int, default=100,
                        help='图片短边最小像素，过小的图片会被过滤')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir.name

    print(f'目标目录: {out_dir.resolve()}')
    print(f'目标数量: {args.max_num}')

    # 优先用百度抓取，失败时继续尝试 Bing
    # 注意：如果目录里已经有文件，编号要从已有数量之后开始，否则会覆盖/跳过
    existing = count_files(out_dir)
    try:
        print('开始百度图片抓取...')
        crawl_baidu(args.keyword_zh, args.max_num, out_dir, file_idx_offset=existing)
        print(f'百度抓取结束，当前文件数: {count_files(out_dir)}')
    except Exception as e:
        print(f'百度抓取失败: {e}')

    current = count_files(out_dir)
    if current < args.max_num:
        try:
            print('开始 Bing 图片补充抓取...')
            crawl_bing(args.keyword_en, args.max_num - current, out_dir,
                       file_idx_offset=current)
            print(f'Bing 抓取结束，当前文件数: {count_files(out_dir)}')
        except Exception as e:
            print(f'Bing 抓取失败: {e}')

    kept = clean_images(out_dir, prefix=prefix, min_side=args.min_side)
    print(f'清洗完成，最终保留 {kept} 张图片，保存在: {out_dir.resolve()}')
    print('注意：抓取结果可能包含插画、卡通或错误图片，建议人工快速浏览一遍再用于评估。')


if __name__ == '__main__':
    main()
