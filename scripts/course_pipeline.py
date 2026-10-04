"""Continue formal experiments automatically once all genuine archives are ready.

This is a project-local job, not a scheduler. Status and errors persist on disk.
The process never marks phone verification complete and does not send messages.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

ROOT=Path(__file__).resolve().parents[1]
EXPLANATIONS={
 'waiting_for_data':('等待完整数据校验','训练与正式结论需要完整、轨迹独立的真实数据','保留后台续传，并以全部归档提取标记作为就绪条件'),
 'indexing':('建立固定轨迹索引','避免相邻帧进入不同集合造成数据泄漏','固定种子42，完整轨迹划分'),
 'training_and_evaluation':('顺序训练三组并做验证/测试评估','比较单帧与时序约束收益，控制精度退化','单卡串行、同更新数、冻结数据索引，验证集选型'),
 'plotting_results':('生成实际训练曲线与指标图','让训练过程和选型依据能够核查','只读取真实日志与指标，保存数据和图像校验值'),
 'comparison_video':('生成四组实际模型对比视频','直观检查动态遮挡、边缘及闪烁','共用固定色标，单独记录推理速度'),
 'building_final_apk':('构建含选定真实模型的APK','手机部署必须使用验证通过的最终权重','复用项目JDK/SDK，执行Kotlin测试后打包'),
 'packaging_software':('打包源码、模型、APK与实验记录','交付应可复现且结果来源明确','包含文件校验值，真机状态独立标注'),
 'software_complete_phone_verification_pending':('等待荣耀100Pro真机验收','电脑测试不能证明手机FPS和生命周期行为','USB授权后安装并持续测量，未测不记完成')}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive-root',default='data/archives')
    p.add_argument('--data-root',default='data/tartanair')
    p.add_argument('--status',default='output/pipeline/status.json')
    p.add_argument('--max-updates',type=int,default=2000)
    p.add_argument('--gpu-budget-hours',type=float,default=11.95,
                   help='Remaining formal GPU experiment budget; default reserves 3 min of the 12 h total for completed smoke/performance checks')
    p.add_argument('--poll-seconds',type=float,default=30)
    a=p.parse_args()
    status=ROOT/a.status;status.parent.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'python':sys.executable,'started_unix':time.time(),'stage':'waiting_for_data',
           'phone_verified':False,'completed':False}
    def write(**kwargs):
        old_stage=state.get('stage')
        state.update(kwargs,updated_unix=time.time())
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');tmp.replace(status)
        progress=status.parent/'progress.md'
        if not progress.exists() or old_stage!=state.get('stage'):
            stage=state.get('stage')
            if stage in EXPLANATIONS:
                what,why,choice=EXPLANATIONS[stage]
                with progress.open('a',encoding='utf-8') as f:
                    f.write(f'\n## {stage}\n\n做什么：{what}。\n\n为什么：{why}。\n\n选择理由：{choice}。\n')
        if state.get('stage')=='packaging_software':
            todo=ROOT/'TODO.md'
            content=todo.read_text(encoding='utf-8')
            for number in (3,6,8,9,10,11):content=content.replace(f'- [ ] {number}.',f'- [x] {number}.')
            content+='\n- 正式模型、视频与APK已完成，实际结果见 `output/experiments/selection.json`；打包状态见 `output/pipeline/status.json`，手机验收仍待连接设备。\n'
            todo.write_text(content,encoding='utf-8')
    def run(stage,command):
        write(stage=stage,command=command)
        print(f'STAGE {stage}: {command}',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    write()
    archive=ROOT/a.archive_root
    try:
        while True:
            manifest_path=archive/'download_manifest.json'
            if manifest_path.exists():
                manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                items=manifest['files']
                ready=[item for item in items if (archive/item['path']).with_suffix('.extracted').exists()]
                write(data_archives_ready=len(ready),data_archives_total=len(items),selected_bytes=manifest['selected_bytes'])
                if len(ready)==len(items):break
            stop=status.parent/'STOP'
            if stop.exists():
                write(stage='stopped_by_user',completed=False);return
            time.sleep(max(a.poll_seconds,1))
        data_manifest=ROOT/a.data_root/'manifest.json'
        run('indexing',[sys.executable,'-B','scripts/prepare_tartanair.py','--root',a.data_root,'--output',str(data_manifest)])
        run('training_and_evaluation',[sys.executable,'-B','-u','scripts/run_course_experiments.py','--manifest',str(data_manifest),
             '--max-updates',str(a.max_updates),'--gpu-budget-hours',str(a.gpu_budget_hours)])
        selection=json.loads((ROOT/'output/experiments/selection.json').read_text(encoding='utf-8'))
        run('plotting_results',[sys.executable,'-B','scripts/plot_course_results.py','--input','output/experiments',
                                '--output','output/curves/formal'])
        run('comparison_video',[sys.executable,'-B','-u','scripts/demo_video.py','--max-frames','300','--device','cuda','--models',
            'original=checkpoints/zipdepth_base_npu.pth',f"single={selection['single_checkpoint']}",
            f"temporal={selection['temporal_candidate_checkpoint']}",f"temporal_smooth={selection['temporal_candidate_checkpoint']}",
            '--smooth-models','temporal_smooth','--output','output/videos/final_comparison.mp4'])
        tools=ROOT/'output/tools/android'
        gradle=tools/'gradle-8.9/bin/gradle.bat'
        jdks=list(tools.glob('jdk-17*'))
        if not gradle.exists() or not jdks:
            write(stage='models_ready_android_tools_pending',completed=False,error='Run android/setup-build.ps1 -Build after tools finish')
            return
        run('building_final_apk',['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass',
                                  '-File','android/build.ps1','-Model','final'])
        destination=ROOT/'output/apk/zipdepth-final-debug.apk'
        if not destination.exists():raise RuntimeError('Build finished without final APK')
        run('packaging_software',[sys.executable,'-B','scripts/package_course.py','--output','output/deliverables/zipdepth-course-software.zip'])
        write(stage='software_complete_phone_verification_pending',completed=False,software_complete=True,
              final_apk=str(destination),selection=str(ROOT/'output/experiments/selection.json'))
    except Exception as error:
        write(stage='failed',error=str(error),traceback=traceback.format_exc(),completed=False)
        raise

if __name__=='__main__':main()
