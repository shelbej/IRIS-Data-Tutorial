from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.listview import ListItemButton
from kivy.uix.label import Label
from kivy.adapters.listadapter import ListAdapter
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty
from kivy.config import Config
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.animation import Animation
#Config.set("input", "mactouch", 'mactouch')

import os
import datetime
import numpy as np
from dateutil.parser import parse
from matplotlib.colors import PowerNorm
from astropy.io import fits
import pickle
from irispy.sji import SJICube
from irispy.spectrograph import IRISSpectrograph




def load_info():
    f = open('/sanhome/shelbe/IRIS/tools/iris_xfiles/recent_searches.pckl', 'rb')
    info = pickle.load(f)
    f.close()
    print('Loaded recent_searches.pckl')

    return info

def save_info(info):
    f = open('/sanhome/shelbe/IRIS/tools/iris_xfiles/recent_searches.pckl', 'wb')
    pickle.dump(info, f)
    f.close()
    print('Saved to recent_searches.pckl')
    show_info(info)

def show_info(info):
    for i in info:
        if 'spatterns' in i:
            print(i)
            for j in info[i]:
                print('     ', j, info[i].get(j))

        else:
            print(i, info.get(i))

info = load_info()
show_info(info)


class edit_spattern_listbutton(ListItemButton):
    pass


class edit_spattern_menu(BoxLayout):
    info = load_info()
    spatterns = info['spatterns']
    def __init__(self, **kwargs):
        super(edit_spattern_menu, self).__init__(**kwargs)
        self.spatterns_list.adapter.bind(on_selection_change=self.selection_changed)
        print(self.spatterns['default'])
    print(spatterns)
    

    name_txtin = ObjectProperty()
    path_txtin = ObjectProperty()
    usetree_check = ObjectProperty()
    subdir_check = ObjectProperty()
    default_check = ObjectProperty()
    spatterns_list = ObjectProperty()
    selected_item = NumericProperty()
    close_button = ObjectProperty()



    def add_spattern(self):
        name = self.ids.name.text
        self.ids.spatterns_listview.adapter.data.extend([name])

        self.spatterns['name'].append(name)
        self.spatterns['path'].append('')
        self.spatterns['usetree'].append(True)
        self.spatterns['searchsubdir'].append(True)
        self.ids.spatterns_listview._trigger_reset_populate()
        print(self.spatterns)

    def delete_spattern(self):
        selection = self.ids.spatterns_listview.adapter.selection[0].text
        self.ids.spatterns_listview.adapter.data.remove(selection)
        del self.spatterns['name'][self.selected_item]
        del self.spatterns['path'][self.selected_item]
        del self.spatterns['usetree'][self.selected_item]
        del self.spatterns['searchsubdir'][self.selected_item]
        self.ids.spatterns_listview._trigger_reset_populate()
        print(self.spatterns)

    def save_spattern(self):
        self.info['spatterns'] = self.spatterns
        save_info(self.info)

        for i in info:
            if 'spatterns' in i:
                print(i)
                for j in info[i]:
                    print('     ', j, info[i].get(j))

            else:
                print(i, info.get(i))

    def set_attribute(self, key, value):
        print(key)
        if 'default' in key:

            if value:
                print('------------------------\nOld Default: ', self.spatterns[key], 'New Default: ',
                      self.selected_item)
                self.spatterns[key] = self.selected_item
        elif 'name' in key:
            print('------------------------\nOld Name: ', self.spatterns[key][self.selected_item], 'New Name: ',
                  value)
            selection = self.ids.spatterns_listview.adapter.selection[0].text
            self.ids.spatterns_listview.adapter.data.remove(selection)
            self.spatterns[key][self.selected_item] = value
            self.ids.spatterns_listview.adapter.data.extend([value])
            self.ids.spatterns_listview._trigger_reset_populate()

        else:
            print('------------------------\nOld value: ', self.spatterns[key][self.selected_item], 'New Value: ', value)
            try:
                self.spatterns[key][self.selected_item] = value
            except:
                print('invalid value')


    def selection_changed(self, *args):
        self.selected_item = self.spatterns['name'].index(args[0].selection[0].text)

    def on_selected_item(self, *args):
        self.ids.name.text = self.spatterns['name'][self.selected_item]
        self.ids.path.text = self.spatterns['path'][self.selected_item]
        self.ids.usetree.active = self.spatterns['usetree'][self.selected_item]
        self.ids.subdir.active = self.spatterns['searchsubdir'][self.selected_item]
        self.ids.default.active = (self.spatterns['default'] == self.selected_item)





class iris_xfiles(BoxLayout):
    info=load_info()
    sdir = info['sdir']
    tend = info['tend']
    tstart = info['tstart']
    spatterns = info['spatterns']
    selected_result_item = ListProperty()
    select = NumericProperty()
    info['filters'] = ''

    def __init__(self, **kwargs):
        super(iris_xfiles, self).__init__(**kwargs)
        self.ids.results_list.adapter.bind(on_selection_change=self.selection_changed)
        self.select=self.spatterns['default']

    def edit_spattern(self):
        edit_popup = Popup(title='Edit Search Patterns',
                           content=edit_spattern_menu(),
                           size_hint=(None, None), size=(400, 400),
                           auto_dismiss=False)
        edit_popup.content.ids.close.bind(on_press=edit_popup.dismiss)
        edit_popup.open()

    def on_dismiss(self):
        self.info = load_info()

    def set_attribute(self, key, value):
        print('------------------------\nOld value: ',info.get(key), 'New Value: ',value)
        if 'tstart' in key or 'tend' in key:
            try:
                info[key] = parse(value)
            except:
                print('invalid date format')
        elif 'sdir' in key:
            self.ids.list.path = self.ids.sdir_input.text
            info[key] = value
        elif 'obsid' or 'filters':
            info[key]=value.replace(' ','').split(',')
        else:
            try:
                info[key] = value
            except:
                print('invalid value')

        save_info(info)
        show_info(info)
        print('Saved: ',info.get(key))

    def load_attribute(self, key):
        if 'tstart' in key or 'tend' in key:
            try:
                value = info.get(key)
                value = value.isoformat()
            except:
                print('invalid date format')
        elif 'obsid' in key or 'filters' in key:
            value = info.get(key)
            value = ', '.join(value)
        else:
            try:
                value = info.get(key)
            except:
                print('invalid key/value')

        print('Loaded: ',key,info.get(key))
        return value

    def select(self,*args):
        file = args[1][0]
        try:
            self.label.text = file
            print(file)


            if 'SJI' in file:
                mc = SJICube(file)
                mc.plot(norm = PowerNorm(.5,0,500))


            elif 'raster' in file:
                nd = IRISSpectrograph(file)
                mg = nd.data['Mg II k 2796']
                mg.index_by_raster[0:,:].plot()
        except:
            pass

    def show(self):
        #TODO: open selected files
        print('Selected files:\n', '\n'.join(self.selected_result_item),'\n ---------------------------------------')

    def startsearch(self):
        info=load_info()
        path = self.ids.sdir_input.text
        usetree = info['spatterns']['usetree'][self.select]
        tdelta = info['tend'] - info['tstart']
        filters = info['filters']
        self.search(info['tstart'], int(tdelta.days+1), obsid=info['obsid'], path=path, usetree=usetree,filters=filters)
        print(self.select,path,usetree,filters)

    def search(self, date, days, obsid='', path='/kale/iris/data/level2/', usetree=True, filters=''):
        if (isinstance(obsid, str) or isinstance(obsid, int)):
            obsid = [str(obsid)]
        results = []
        if usetree:
            for d in range(0, days):
                print(date)

                try:
                    for root, dir, file in os.walk(path + date.strftime('%Y/%m/%d')):
                        for obs in obsid:
                            if str(obs) in dir[:]:
                                for i in file:
                                    for f in filters:
                                        if (f in i) and (i.endswith(".fits")):
                                            results.append(os.path.join(root, i))
                except:
                    print('Valid directory: ',os.path.isdir(path + date.strftime('%Y/%m/%d')),
                          root, i, f, obs, f in i , obs in i)
                date += datetime.timedelta(days=1)
        else:
            for file in os.listdir(path):
                if file.endswith(".fits"):
                    results.append(os.path.join(path, file))

        results.sort()
        result_headers = []
        for r in results:
            header = fits.getheader(r, ext=0)
            header_string = header.get('STARTOBS') + '    ' + header.get('OBSID') + '    ' + header.get(
                'OBS_DESC') + '    ' + str(header.get('XCEN')) + '    ' + str(header.get('YCEN')) + '    ' + str(
                round(header.get('SAT_ROT'),2))
            result_headers.append(header_string)
        self.ids.results_list.item_strings = [h for h in result_headers]
        print('Contains ', len(results), ' files')
        return results
        # Todo: list adapter to add/remove
        save_info(info)

    def selection_changed(self, *args):
        self.selected_result_item=[]
        for s in args[0].selection:
            self.selected_result_item.append(s.text)

    def on_selected_item(self, *args):
        pass

    def show_selected_value(self, text):
        self.select = self.spatterns['name'].index(text)
        self.ids.sdir_input.text = self.spatterns['path'][self.select]



class iris_xfilesApp(App):

    def build(self):
        #self.load_kv('iris_xfiles.kv')
        return iris_xfiles()#edit_spattern_menu()#

if __name__ == '__main__':
    iris_xfilesApp().run()


    #     try: self.label.text = results
    #     except: pass




    # dropdown = DropDown()
    #
    # for i, txt in enumerate(spatterns['name']):
    #
    #     btn = Button(text=txt, size_hint_y=None, height=44)
    #
    #     btn.bind(on_release= lambda btn: dropdown.select(btn.text))
    #     dropdown.add_widget(btn)
    #
    # dropdown.bind(on_select=lambda instance, x: setattr(self.ids.sel_spattern, 'text', x))