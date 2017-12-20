#coding-utf-8
from config import Config


class ActionManipulator:
    def __init__(self):
        config = Config()

    def get_action_templates(self):
        return []

    def greeting(self):
        return u'你好'

    def closing(self):
        return u'再见'

    def request_service_name(self):

        return u'请问具体是哪一种业务呢'

    def request_service_type(self):

        return u'请问有什么能帮助您的'

    def request_operation(self):

        return u'请问您要进行怎样的操作呢'

    def request_confirmation(self, stype, sname, operation):

        return u'请确认您的操作请求: {} - {} - {}'.format(stype, sname, operation)

    def retrieve_info(self, sname):

        return u'以下是关于{}的全部信息'.format(sname)

    def cancel(self):

        return u'已为您取消'

    def apology_name(self):

        return u'对不起, 没有找到相关业务名称'

    def apology_type(self):

        return u'对不起, 无法为您提供相关服务'

    def confirm(self):

        return u'好的, 这就为您操作'
