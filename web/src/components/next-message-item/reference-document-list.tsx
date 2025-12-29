import { Card, CardContent } from '@/components/ui/card';
import { Docagg } from '@/interfaces/database/chat';
import FileIcon from '../file-icon';

export function ReferenceDocumentList({ list }: { list: Docagg[] }) {
  return (
    <section className="flex gap-3 flex-wrap">
      {list.map((item) => (
        <Card key={item.doc_id} className="cursor-default">
          <CardContent className="p-2 space-x-2 flex items-center">
            <FileIcon id={item.doc_id} name={item.doc_name}></FileIcon>
            <span
              className="text-text-sub-title-invert"
              style={{ wordBreak: 'break-all' }}
            >
              {item.doc_name}
            </span>
          </CardContent>
        </Card>
      ))}
    </section>
  );
}
