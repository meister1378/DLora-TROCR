from ftplib import FTP
import os
from PIL import Image

def find_correct_y_path():
    """OCR 데이터셋에서 이미지 파일을 찾고 권한 테스트"""
    
    try:
        # FTP 연결
        ftp = FTP()
        ftp.encoding = 'utf-8'
        ftp.connect('172.30.1.226', 21, timeout=10)
        ftp.login('admin', 'admin')
        ftp.set_pasv(True)
        
        print("✅ FTP 연결 성공")
        
        # OCR 데이터셋 경로로 직접 이동
        ocr_path = "Y:\\ocr_dataset\\공공행정문서 OCR\\Validation\\[원천]validation\\02.원천데이터(Jpg)\\농림.축산지원\\5350034"
        print(f"\n📁 OCR 데이터셋 경로로 이동: {ocr_path}")
        
        try:
            ftp.cwd(ocr_path)
            print("✅ OCR 경로 접근 성공!")
            
            # 현재 위치 확인
            current_path = ftp.pwd()
            print(f"현재 위치: {current_path}")
            
            # 폴더 내용 확인
            print(f"\n📂 OCR 폴더 내용:")
            ocr_files = []
            ftp.retrlines('LIST', ocr_files.append)
            
            for line in ocr_files:
                print(f"   {line}")
            
            # 하위 폴더들 찾기
            subfolders = []
            for line in ocr_files:
                parts = line.split()
                if len(parts) >= 9 and parts[0].startswith('d'):
                    folder_name = ' '.join(parts[8:])
                    if folder_name not in ['.', '..']:
                        subfolders.append(folder_name)
            
            print(f"\n🔍 하위 폴더들: {subfolders}")
            
            # 각 하위 폴더에서 이미지 검색 (경로 문제 해결)
            all_images = []
            for subfolder in subfolders:
                try:
                    print(f"\n📁 {subfolder} 폴더 확인...")
                    # 현재 위치에서 상대 경로로 이동
                    ftp.cwd(subfolder)
                    
                    sub_files = []
                    ftp.retrlines('LIST', sub_files.append)
                    
                    print(f"   📄 {subfolder} 내용 ({len(sub_files)}개 항목):")
                    
                    images_in_folder = 0
                    for file_line in sub_files:
                        parts = file_line.split()
                        if len(parts) >= 9:
                            filename = ' '.join(parts[8:])
                            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                                size = parts[4] if len(parts) > 4 else 'unknown'
                                print(f"      🖼️  {filename} ({size} bytes)")
                                all_images.append({
                                    'subfolder': subfolder,
                                    'filename': filename,
                                    'size': size
                                })
                                images_in_folder += 1
                            else:
                                # 일반 파일도 표시 (처음 3개만)
                                if len([f for f in sub_files if not f.split()[-1].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]) < 4:
                                    print(f"      📄 {filename}")
                    
                    print(f"   ✅ {images_in_folder}개 이미지 파일 발견!")
                    
                    # 부모 디렉토리로 돌아가기
                    ftp.cwd('..')
                    
                except Exception as e:
                    print(f"   ❌ {subfolder} 접근 실패: {e}")
                    # 에러 발생시 원래 위치로 돌아가기
                    try:
                        ftp.cwd(ocr_path)
                    except:
                        pass
            
            # 이미지 다운로드 및 처리 테스트
            if all_images:
                print(f"\n🎉 총 {len(all_images)}개의 이미지 파일 발견!")
                
                # 첫 번째 이미지 다운로드 테스트
                test_image = all_images[0]
                print(f"\n📥 다운로드 테스트: {test_image['filename']}")
                print(f"   위치: {test_image['subfolder']} 폴더")
                
                try:
                    # OCR 폴더로 돌아간 후 하위 폴더로 이동
                    print(f"   현재 위치에서 {test_image['subfolder']} 폴더로 이동...")
                    ftp.cwd(test_image['subfolder'])
                    print(f"   이동 성공! 현재 위치: {ftp.pwd()}")
                    
                    local_filename = f"ocr_{test_image['filename']}"
                    with open(local_filename, 'wb') as f:
                        ftp.retrbinary(f"RETR {test_image['filename']}", f.write)
                    
                    downloaded_size = os.path.getsize(local_filename)
                    print(f"✅ 다운로드 성공: {downloaded_size:,} bytes")
                    
                    # 이미지 처리 권한 테스트
                    print("\n🔧 이미지 처리 권한 테스트:")
                    
                    try:
                        # 1. 이미지 읽기
                        with Image.open(local_filename) as img:
                            print(f"✅ 이미지 읽기 성공: {img.size} pixels, {img.mode} mode")
                            original_img = img.copy()
                        
                        # 2. 이미지 변환들
                        if original_img.mode != 'RGB':
                            rgb_img = original_img.convert('RGB')
                        else:
                            rgb_img = original_img
                        
                        # 3. 리사이즈
                        resized = rgb_img.resize((200, 200))
                        resize_name = f"ocr_resized_{test_image['filename']}"
                        resized.save(resize_name, 'JPEG')
                        resize_size = os.path.getsize(resize_name)
                        print(f"✅ 리사이즈 성공: {resize_name} ({resize_size:,} bytes)")
                        
                        # 4. 그레이스케일 변환
                        gray = rgb_img.convert('L')
                        gray_name = f"ocr_gray_{test_image['filename']}"
                        gray.save(gray_name, 'JPEG')
                        gray_size = os.path.getsize(gray_name)
                        print(f"✅ 그레이스케일 변환 성공: {gray_name} ({gray_size:,} bytes)")
                        
                        # 5. 썸네일 생성
                        thumb_img = rgb_img.copy()
                        thumb_img.thumbnail((100, 100))
                        thumb_name = f"ocr_thumb_{test_image['filename']}"
                        thumb_img.save(thumb_name, 'JPEG')
                        thumb_size = os.path.getsize(thumb_name)
                        print(f"✅ 썸네일 생성 성공: {thumb_name} ({thumb_size:,} bytes)")
                        
                        print(f"\n🎊 Y 드라이브 OCR 데이터 FTP 이미지 처리 권한 확인 완료!")
                        print("=" * 50)
                        print("✅ 읽기 권한: 가능 (FTP에서 이미지 다운로드)")
                        print("✅ 불러오기 권한: 가능 (PIL로 이미지 로드)") 
                        print("✅ 변환 권한: 가능 (리사이즈, 그레이스케일)")
                        print("✅ 생성 권한: 가능 (새 이미지 파일 저장)")
                        print("=" * 50)
                        
                    except Exception as e:
                        print(f"❌ 이미지 처리 실패: {e}")
                        
                except Exception as e:
                    print(f"❌ 다운로드 실패: {e}")
            
            else:
                print("\n❌ 이미지 파일을 찾을 수 없습니다.")
                print("💡 다른 OCR 데이터 경로를 확인해보세요.")
            
        except Exception as e:
            print(f"❌ OCR 경로 접근 실패: {e}")
        
        ftp.quit()
        print("\n✅ FTP 연결 종료")
        
    except Exception as e:
        print(f"❌ FTP 연결 실패: {e}")

if __name__ == "__main__":
    print("=== Y 드라이브 OCR 데이터셋 이미지 권한 테스트 ===")
    find_correct_y_path() 